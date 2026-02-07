"""
Trivalaya Visual Search - FastAPI
Upload a coin image, get the K most similar coins from the dataset.

Uses CLIP ViT-B-32 (512-d) for query embedding, matched against
pre-derived 512-d search features from paired obv+rev CLIP vectors.
Metadata comes from the enriched clustering CSV (23k+ coins).
"""

# =============================================================================
# visual_search/app2.py
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
from io import BytesIO
from typing import Optional
from dataclasses import dataclass
import sys
import os
import cv2
cv2.setNumThreads(1)  # stabilize on 1vCPU
import open_clip
import json
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import torch
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
import html as html_module
import threading

# -----------------------------------------------------------------------------
# 1. Path Safety: Insert project root at the START of sys.path
# -----------------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# -----------------------------------------------------------------------------
# 2. Optional Imports
# -----------------------------------------------------------------------------
try:
    _vision_src = Path("/root/trivalaya-vision/src")
    if str(_vision_src) not in sys.path:
        sys.path.insert(0, str(_vision_src))
    from two_coin_resolver import TwoCoinResolver
    TWO_COIN_AVAILABLE = True
except ImportError:
    print("⚠️ TwoCoinResolver not found at /root/trivalaya-vision/src. Auto-splitting disabled.")
    TWO_COIN_AVAILABLE = False
    class TwoCoinResolver: pass
except Exception as e:
    print(f"⚠️ Error importing TwoCoinResolver: {e}")
    TWO_COIN_AVAILABLE = False
    class TwoCoinResolver: pass

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # 512-d search features derived from paired CLIP embeddings
    "features_path": "cluster_output_clip_spaces/search_features_512.npy",
    "metadata_path": "cluster_output_clip_spaces/cluster_results.enriched.csv",

    # CLIP Configuration (512-dim ViT-B-32)
    "clip_model": "ViT-B-32",
    "clip_pretrained": "laion2b_s34b_b79k",

    "default_k": 10,
    "max_k": 50,

    # Period Classifier (Optional - set paths if available, else ignored gracefully)
    "period_model_path": "trivalaya_model_v5.pth",
    "period_meta_path": "trivalaya_v6_pairs_meta.json",

    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # DigitalOcean Spaces - public base URL for crop images
    "spaces_base_url": "https://trivalaya-data.sfo3.digitaloceanspaces.com",
}

# =============================================================================
# Helpers
# =============================================================================

def _safe_str(v) -> str:
    if v is None: return ""
    try:
        if isinstance(v, float) and np.isnan(v): return ""
    except Exception: pass
    return "" if str(v).lower() == "nan" else str(v)

def _json_safe(obj):
    import numpy as _np
    if obj is None: return None
    if isinstance(obj, (_np.generic,)): return obj.item()
    if isinstance(obj, _np.ndarray): return obj.tolist()
    if isinstance(obj, (list, tuple)): return [_json_safe(x) for x in obj]
    if isinstance(obj, dict): return {str(k): _json_safe(v) for k, v in obj.items()}
    return obj

def _bytes_to_bgr(contents: bytes) -> np.ndarray:
    arr = np.frombuffer(contents, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return bgr

def _prep_gray_and_binary(bgr: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    blurred = cv2.GaussianBlur(g, (7, 7), 0)
    _, bin_a = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin_b = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    def fg_ratio(m): return cv2.countNonZero(m) / (m.shape[0] * m.shape[1])
    ra, rb = fg_ratio(bin_a), fg_ratio(bin_b)
    def score(r): return 10.0 if (r < 0.03 or r > 0.80) else abs(r - 0.40)
    binary = bin_a if score(ra) <= score(rb) else bin_b
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return g, binary

def _candidate_bbox_from_binary(binary: np.ndarray):
    try:
        h, w = binary.shape[:2]
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        c = max(cnts, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        mx, my = int(bw * 0.05), int(bh * 0.05)
        x = max(0, x - mx); y = max(0, y - my)
        bw = min(w - x, bw + 2 * mx); bh = min(h - y, bh + 2 * my)
        return (int(x), int(y), int(bw), int(bh))
    except Exception: return None

def build_lot_url(auction_house: str, sale_id, lot_number) -> str:
    house = _safe_str(auction_house).lower().strip()
    sale, lot = _safe_str(sale_id).strip(), _safe_str(lot_number).strip()
    if not house or not lot: return ""
    try: lot_int = int(lot)
    except Exception: lot_int = None
    if house == "numisbids": return f"https://www.numisbids.com/n.php?p=lot&sid={sale}&lot={lot}"
    if house == "leu": return f"https://leunumismatik.com/en/lot/{sale}/{lot}"
    if house == "gorny": return f"https://auktionen.gmcoinart.de/Los/{sale}/{lot}.0"
    if house == "nomos": return f"https://nomosag.com/nomos-{sale}/{lot}"
    if house == "obolos": return f"https://nomosag.com/obolos-{sale}/{lot}"
    if house == "cng": return f"https://cngcoins.com/Lot.aspx?LOT_ID={lot}"
    if house == "spink" and lot_int: return f"https://www.spink.com/lot/{sale}{lot_int:06d}"
    if house == "kuenker": return f"https://www.kuenker.de/en/archiv/stueck/{lot}"
    if house == "kuenker_auex": return f"https://auex.de/de/product/{sale}-{lot}"
    return ""

def _spaces_url(image_path: str) -> str:
    path = _safe_str(image_path).strip()
    if not path or path.startswith("http"): return path
    path = path.lstrip("/")
    return f"{CONFIG['spaces_base_url']}/{path}"

# =============================================================================
# Period Classifier Logic
# =============================================================================

def load_period_classifier(model_path: str, meta_path: str, device: str):
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        print("⚠️ Period classifier files not found. Reranking disabled.")
        return None, None, [], 0

    with open(meta_path) as f:
        meta = json.load(f)

    period_to_idx = meta["period_to_idx"]
    idx_to_period = meta["idx_to_period"]
    num_classes = len(period_to_idx)

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(1280, num_classes)

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.to(device).eval()
        print("✅ Period classifier loaded successfully.")
        return model, period_to_idx, idx_to_period, num_classes
    except Exception as e:
        print(f"⚠️ Failed to load period classifier: {e}")
        return None, None, [], 0

def get_period_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def classify_period_probs(model, img, device, transform) -> np.ndarray:
    if model is None: return np.array([])
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy().flatten()

def build_period_idx_array(metadata_df, period_to_idx: dict) -> np.ndarray:
    if not period_to_idx: return np.full(len(metadata_df), -1)
    def canon(x): return ("" if x is None else str(x)).strip().lower()
    return np.array([
        period_to_idx.get(canon(row.get("period")), -1)
        for _, row in metadata_df.iterrows()
    ], dtype=np.int32)

# =============================================================================
# Search Engine
# =============================================================================

@dataclass
class SearchEngine:
    features: np.ndarray
    metadata: pd.DataFrame
    model: torch.nn.Module
    preprocess: callable
    device: str
    period_model: Optional[torch.nn.Module]
    period_transform: callable
    period_to_idx: dict
    idx_to_period: list
    period_idx_per_row: np.ndarray

    def embed_image(self, img: Image.Image) -> np.ndarray:
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(img_tensor)
            embedding = embedding / (embedding.norm(dim=-1, keepdim=True) + 1e-12)
        return embedding.cpu().numpy().flatten()

    def search_with_period_boost(self, query_img, k=10, period_filter=None, denom_filter=None):
        # 1. Embed Query
        query_emb = self.embed_image(query_img)

        # 2. Cosine Similarity
        sim = self.features @ query_emb

        # 3. Period Reranking (if model available)
        if self.period_model:
            Pq = classify_period_probs(self.period_model, query_img, self.device, self.period_transform)
            conf = float(Pq.max())
            lam = 0.04 + 0.10 * max(0.0, conf - 0.5) / 0.5

            # Identify query period
            q_idx = int(Pq.argmax())
            q_period_name = self.idx_to_period[q_idx]
        else:
            lam = 0.0
            q_period_name = "unknown"
            conf = 0.0

        # 4. Filter & Score
        indices = np.argsort(-sim)
        results = []

        baseline = 1.0 / len(self.idx_to_period) if self.idx_to_period else 0.0

        for idx in indices:
            row = self.metadata.iloc[idx]

            # Apply Period Filter
            r_period = _safe_str(row.get("parser_period", row.get("period", ""))).lower()
            if period_filter and period_filter != "all" and period_filter.lower() not in r_period:
                continue

            # Apply Denomination Filter
            if denom_filter and denom_filter != "all":
                r_denom = _safe_str(row.get("denomination", "")).lower()
                if denom_filter.lower() not in r_denom:
                    continue

            # Compute Final Score
            final_score = float(sim[idx])
            boost_val = 0.0

            if lam > 0:
                pidx = int(self.period_idx_per_row[idx])
                if pidx >= 0:
                    p_score = float(Pq[pidx])
                    boost_val = lam * (p_score - baseline)
                    final_score += boost_val

            # Format Response — use enriched CSV columns
            obv_url = _spaces_url(_safe_str(row.get("obv_path_x", "")))
            rev_url = _spaces_url(_safe_str(row.get("rev_path_x", "")))
            period = _safe_str(row.get("parser_period", row.get("period", "unknown")))
            denomination = _safe_str(row.get("denomination", ""))
            material = _safe_str(row.get("material", ""))
            lot_title = _safe_str(row.get("lot_title", ""))
            lot_url = _safe_str(row.get("lot_url", ""))
            if not lot_url:
                lot_url = build_lot_url(
                    row.get("auction_house", ""),
                    row.get("sale_id", ""),
                    row.get("lot_number", ""),
                )
            auction_house = _safe_str(row.get("auction_house", ""))
            cluster_id = _safe_str(row.get("cluster_id", ""))

            results.append({
                "id": int(row.get("id", row.get("coin_id", idx))),
                "obv_url": obv_url,
                "rev_url": rev_url,
                "period": period,
                "denomination": denomination,
                "material": material,
                "similarity": float(sim[idx]),
                "cosine": f"{float(sim[idx]):.3f}",
                "final_score": final_score,
                "period_boost": boost_val,
                "query_period": q_period_name,
                "query_period_conf": conf,
                "lot_title": lot_title,
                "lot_url": lot_url,
                "auction_house": auction_house,
                "cluster_id": cluster_id,
            })

            if len(results) >= k: break

        return results

# =============================================================================
# Loader
# =============================================================================

def load_search_engine() -> SearchEngine:
    print(f"🔄 Loading CLIP model {CONFIG['clip_model']}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CONFIG["clip_model"], pretrained=CONFIG["clip_pretrained"], device=CONFIG["device"]
    )
    model.eval()

    if not os.path.exists(CONFIG["features_path"]):
        raise FileNotFoundError(f"❌ Missing features: {CONFIG['features_path']}")

    print(f"🔄 Loading features from {CONFIG['features_path']}...")
    features = np.load(CONFIG["features_path"])
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)

    print(f"🔄 Loading metadata from {CONFIG['metadata_path']}...")
    metadata = pd.read_csv(CONFIG["metadata_path"])

    # Validation
    if len(features) != len(metadata):
        raise ValueError(f"❌ Row mismatch: {len(features)} features vs {len(metadata)} metadata rows")

    feat_dim = features.shape[1]
    dummy_tensor = preprocess(Image.new("RGB", (224,224))).unsqueeze(0).to(CONFIG["device"])
    with torch.no_grad(): out = model.encode_image(dummy_tensor)
    model_dim = out.shape[-1]

    if feat_dim != model_dim:
        raise ValueError(f"❌ Dimension Mismatch! DB: {feat_dim}d, Model: {model_dim}d")

    # Load Period Classifier
    p_model, p_to_idx, idx_to_p, n_cls = load_period_classifier(
        CONFIG["period_model_path"], CONFIG["period_meta_path"], CONFIG["device"]
    )
    p_idx_arr = build_period_idx_array(metadata, p_to_idx)

    print(f"✅ Engine Ready: {len(features)} coins loaded.")
    return SearchEngine(features, metadata, model, preprocess, CONFIG["device"],
                        p_model, get_period_transform(), p_to_idx, idx_to_p, p_idx_arr)

# =============================================================================
# FastAPI Routes
# =============================================================================

app = FastAPI(title="Trivalaya", version="0.7.0")
engine: SearchEngine = None
search_lock = threading.Lock()

@app.on_event("startup")
async def startup():
    global engine
    engine = load_search_engine()

@app.get("/", response_class=HTMLResponse)
async def home():
    periods = []
    denoms = []
    if engine and not engine.metadata.empty:
        periods = sorted(
            engine.metadata["parser_period"]
            .fillna(engine.metadata.get("period", pd.Series()))
            .dropna().astype(str).unique()
        )
        denoms = sorted(
            engine.metadata["denomination"].dropna().astype(str).unique()
        )

    p_opts = '<option value="all">All Periods</option>' + "".join(
        f'<option value="{html_module.escape(p)}">{html_module.escape(p)}</option>'
        for p in periods if p
    )
    d_opts = '<option value="all">All Denominations</option>' + "".join(
        f'<option value="{html_module.escape(str(d))}">{html_module.escape(str(d))}</option>'
        for d in denoms if d
    )
    return HTML_TEMPLATE.replace("{{PERIOD_OPTIONS}}", p_opts).replace("{{DENOM_OPTIONS}}", d_opts)

@app.post("/search")
def search_endpoint(
    file: UploadFile = File(...),
    k: int = Query(10, ge=1, le=50),
    period: Optional[str] = None,
    denomination: Optional[str] = None,
    split: bool = Query(True)
):
    if not search_lock.acquire(blocking=False):
        raise HTTPException(429, "Server busy")
    try:
        contents = file.file.read()
        if not contents: raise HTTPException(400, "Empty file")

        def run_query(pil):
            return engine.search_with_period_boost(pil, k=k, period_filter=period, denom_filter=denomination)

        # Split Logic — guarded by cheap heuristics from app.py v1
        if split and TWO_COIN_AVAILABLE:
            try:
                bgr = _bytes_to_bgr(contents)
                h, w = bgr.shape[:2]

                # Gate 1: aspect ratio — side-by-side coins are wider than tall
                if w / h > 1.2:
                    gray, binary = _prep_gray_and_binary(bgr)
                    bb_fast = _candidate_bbox_from_binary(binary)
                    candidate_bbox = (0, 0, w, h)
                    do_resolve = True

                    # Gate 2: blob shape — skip resolver if blob looks like a single coin
                    if bb_fast:
                        bx, by, bw, bh = bb_fast
                        blob_ar = (bw / bh) if bh else 0.0
                        blob_area_ratio = (bw * bh) / float(h * w)
                        if blob_ar < 1.15 or blob_area_ratio < 0.06:
                            do_resolve = False
                        else:
                            candidate_bbox = bb_fast

                    if do_resolve:
                        try:
                            res = TwoCoinResolver().resolve(bgr, binary, gray, candidate_bbox)
                        except TypeError:
                            res = TwoCoinResolver().resolve(bgr, binary, gray)
                        if res.get("status") == "split" and len(res.get("coins", [])) == 2:
                            return _json_safe({
                                "split_status": "split",
                                "coins": [
                                    {"side": c["side"], "results": run_query(Image.fromarray(cv2.cvtColor(c["crop"], cv2.COLOR_BGR2RGB)))}
                                    for c in res["coins"]
                                ]
                            })
            except Exception as e:
                print(f"Split error: {e}")

        # Fallback: Single Image Search
        pil = Image.open(BytesIO(contents)).convert("RGB")
        return _json_safe({"split_status": "single", "results": run_query(pil)})

    finally:
        search_lock.release()

@app.get("/stats")
def stats_endpoint():
    if not engine:
        raise HTTPException(503, "Engine not loaded")
    md = engine.metadata
    period_col = md["parser_period"].fillna(md.get("period", pd.Series()))
    return {
        "total_coins": len(md),
        "feature_dim": int(engine.features.shape[1]),
        "period_distribution": period_col.value_counts().to_dict(),
        "denomination_distribution": md["denomination"].dropna().value_counts().to_dict(),
        "auction_house_distribution": md["auction_house"].dropna().value_counts().to_dict(),
    }

# =============================================================================
# HTML Template
# =============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trivalaya</title>
    <style>
        :root { --bg: #0d0d0d; --card: #141414; --text: #e8e4dc; --accent: #c9a227; }
        body { font-family: serif; background: var(--bg); color: var(--text); margin: 0; padding: 2rem; }
        .container { max-width: 1400px; margin: 0 auto; display: grid; grid-template-columns: 300px 1fr; gap: 2rem; }
        .upload-zone { border: 2px dashed #333; padding: 2rem; text-align: center; cursor: pointer; margin-bottom: 1rem; }
        .upload-zone:hover { border-color: var(--accent); }
        .upload-zone img { max-width: 100%; display: none; }
        .controls label { display: block; margin: 1rem 0 0.5rem; color: #888; font-size: 0.9rem; }
        select, input, button { width: 100%; padding: 0.8rem; background: var(--card); border: 1px solid #333; color: white; }
        button { background: var(--accent); color: black; font-weight: bold; cursor: pointer; margin-top: 1.5rem; }
        button:disabled { opacity: 0.5; cursor: wait; }

        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 1.5rem; }
        .card { background: var(--card); border: 1px solid #333; position: relative; transition: transform 0.2s; }
        .card:hover { transform: translateY(-3px); border-color: var(--accent); }
        .card .thumbs { display: flex; gap: 2px; background: #000; }
        .card .thumbs img { flex: 1; min-width: 0; aspect-ratio: 1; object-fit: contain; }
        .info { padding: 1rem; }
        .score { color: var(--accent); font-family: monospace; font-size: 1.2rem; }
        .meta { font-size: 0.85rem; color: #888; margin-top: 0.5rem; }
        .badge { position: absolute; top: 0; right: 0; background: var(--accent); color: black; font-size: 0.7rem; padding: 2px 6px; font-weight: bold; z-index: 1; }
        .checkbox-row { display: flex; align-items: center; gap: 0.5rem; margin-top: 1rem; }
        .checkbox-row input { width: auto; }
        .checkbox-row label { margin: 0; color: var(--text); }
    </style>
</head>
<body>
    <div class="container">
        <aside>
            <h1 style="color:var(--accent); margin-top:0;">TRIVALAYA</h1>
            <div class="upload-zone" id="dropZone">
                <img id="preview">
                <p class="placeholder">Drop coin image here</p>
            </div>
            <input type="file" id="fileInput" hidden>

            <div class="controls">
                <label>Results (K)</label>
                <input type="number" id="kVal" value="10" min="1" max="50">
                <label>Period</label>
                <select id="pFilter">{{PERIOD_OPTIONS}}</select>
                <label>Denomination</label>
                <select id="dFilter">{{DENOM_OPTIONS}}</select>
                <div class="checkbox-row">
                    <input type="checkbox" id="splitCheck" checked>
                    <label for="splitCheck">Auto-split Obv/Rev</label>
                </div>
                <button id="btn">Search</button>
            </div>
        </aside>
        <main>
            <div id="status" style="margin-bottom: 1rem; color: #888;"></div>
            <div class="grid" id="results"></div>
        </main>
    </div>

    <script>
        const drop = document.getElementById('dropZone');
        const fileIn = document.getElementById('fileInput');
        const btn = document.getElementById('btn');
        let file = null;

        drop.onclick = () => fileIn.click();
        fileIn.onchange = e => handleFile(e.target.files[0]);
        drop.ondragover = e => { e.preventDefault(); drop.style.borderColor = '#c9a227'; };
        drop.ondrop = e => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); };

        function handleFile(f) {
            if(!f) return;
            file = f;
            const r = new FileReader();
            r.onload = e => {
                document.getElementById('preview').src = e.target.result;
                document.getElementById('preview').style.display = 'block';
                document.querySelector('.placeholder').style.display = 'none';
            };
            r.readAsDataURL(f);
        }

        btn.onclick = async () => {
            if(!file) return alert('Please upload an image first');
            btn.disabled = true;
            document.getElementById('status').innerText = 'Processing...';
            document.getElementById('results').innerHTML = '';

            const fd = new FormData();
            fd.append('file', file);

            const k = document.getElementById('kVal').value;
            const p = document.getElementById('pFilter').value;
            const d = document.getElementById('dFilter').value;
            const doSplit = document.getElementById('splitCheck').checked;

            let qs = `?k=${k}&split=${doSplit}`;
            if(p !== 'all') qs += `&period=${encodeURIComponent(p)}`;
            if(d !== 'all') qs += `&denomination=${encodeURIComponent(d)}`;

            try {
                const res = await fetch('/search'+qs, {method: 'POST', body: fd});
                if(!res.ok) throw new Error(await res.text());
                const data = await res.json();
                render(data);
            } catch(e) {
                alert('Error: ' + e.message);
            } finally {
                btn.disabled = false;
            }
        };

        function render(data) {
            const grid = document.getElementById('results');
            let items = [];
            let statusText = "";

            if (data.split_status === 'split') {
                statusText = 'Split Successful.';
                data.coins.forEach(c => {
                    c.results.forEach(r => {
                        r._side = c.side;
                        items.push(r);
                    });
                });
            } else {
                statusText = `Found ${data.results.length} matches.`;
                items = data.results;
            }
            document.getElementById('status').innerText = statusText;

            items.forEach(r => {
                const div = document.createElement('div');
                div.className = 'card';

                const obvUrl = r.obv_url || '';
                const revUrl = r.rev_url || '';
                const badge = r._side ? `<div class="badge">${r._side.toUpperCase()}</div>` : '';
                const lotLink = r.lot_url
                    ? `<a href="${r.lot_url}" target="_blank" style="color:#c9a227">View Lot</a>`
                    : '';
                const metaParts = [r.period, r.denomination, r.material].filter(Boolean);

                div.innerHTML = `
                    ${badge}
                    <div class="thumbs">
                        ${obvUrl ? `<img src="${obvUrl}" loading="lazy" title="Obverse">` : ''}
                        ${revUrl ? `<img src="${revUrl}" loading="lazy" title="Reverse">` : ''}
                    </div>
                    <div class="info">
                        <div class="score">${r.cosine}</div>
                        <div class="meta">${metaParts.join(' &middot; ')}</div>
                        <div style="font-size:0.9rem; margin-top:0.5rem; font-weight:bold;">${r.lot_title || 'Untitled'}</div>
                        <div style="margin-top:0.5rem;">${lotLink}</div>
                    </div>
                `;
                grid.appendChild(div);
            });
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
