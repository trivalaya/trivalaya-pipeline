"""
Trivalaya Visual Search - FastAPI
Upload a coin image, get the K most similar coins from the dataset.
Features:
- Robust Obverse/Reverse auto-splitting (using OpenCV + Watershed)
- CLIP-based semantic search
- Secure file serving (Allowed directories only)
- Interactive UI with filters

Usage:
    python visual_search/app.py
"""

# =============================================================================
# visual_search/app.py 
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
import torch
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import html as html_module
import threading
from contextlib import contextmanager

# -----------------------------------------------------------------------------
# 1. Path Safety: Insert project root at the START of sys.path
# -----------------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# -----------------------------------------------------------------------------
# 2. Optional Imports (Layer1 & TwoCoinResolver)
# -----------------------------------------------------------------------------
# Optional Layer1 structural salience
try:
    # Try importing from src first (common structure)
    try:
        from src.layer1_structural_salience import layer_1_structural_salience
    except ImportError:
        from layer1_structural_salience import layer_1_structural_salience
    LAYER1_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Layer1 unavailable: {e}")
    LAYER1_AVAILABLE = False

# Optional two-coin resolver
try:
    from two_coin_resolver import TwoCoinResolver
    TWO_COIN_AVAILABLE = True
except ImportError:
    print("⚠️ TwoCoinResolver not found. Auto-splitting disabled.")
    TWO_COIN_AVAILABLE = False
    # Define a dummy class to prevent NameError in Pylance/Runtime
    class TwoCoinResolver: pass 
except Exception as e:
    print(f"⚠️ Error importing TwoCoinResolver: {e}")
    TWO_COIN_AVAILABLE = False
    class TwoCoinResolver: pass

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "features_path": "cluster_output/features.npy",
    "metadata_path": "cluster_output/cluster_results.enriched.csv",
    "allowed_dirs": ["trivalaya_data", "cluster_output"], 
    
    # CLIP Configuration (512-dim ViT-B-32)
    "clip_model": "ViT-B-32",
    "clip_pretrained": "laion2b_s34b_b79k",
    
    "default_k": 10,
    "max_k": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
# ... (Keep your existing Helpers: _safe_str, _json_safe, _bytes_to_bgr, etc.) ...
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
    """Recursively convert numpy / torch scalar-like objects to JSON-safe Python types."""
    import numpy as _np
    if obj is None:
        return None
    if isinstance(obj, (_np.generic,)):
        return obj.item()
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    try:
        import torch as _torch
        if isinstance(obj, _torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    return obj

def _bytes_to_bgr(contents: bytes) -> np.ndarray:
    """Convert uploaded bytes to OpenCV BGR image with validation."""
    arr = np.frombuffer(contents, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return bgr

def _prep_gray_and_binary(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prepare grayscale and binary mask for splitting."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    blurred = cv2.GaussianBlur(g, (7, 7), 0)

    _, bin_a = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin_b = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def fg_ratio(m): return cv2.countNonZero(m) / (m.shape[0] * m.shape[1])
    ra, rb = fg_ratio(bin_a), fg_ratio(bin_b)

    def score(r):
        if r < 0.03 or r > 0.80: return 10.0
        return abs(r - 0.40)

    binary = bin_a if score(ra) <= score(rb) else bin_b
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return g, binary

def _candidate_bbox_from_binary(binary: np.ndarray):
    """Fast bbox from largest contour in binary mask."""
    try:
        h, w = binary.shape[:2]
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        mx = int(bw * 0.05)
        my = int(bh * 0.05)
        x = max(0, x - mx); y = max(0, y - my)
        bw = min(w - x, bw + 2 * mx); bh = min(h - y, bh + 2 * my)
        return (int(x), int(y), int(bw), int(bh))
    except Exception:
        return None

def build_lot_url(auction_house: str, sale_id, lot_number) -> str:
    house = _safe_str(auction_house).lower().strip()
    sale, lot = _safe_str(sale_id).strip(), _safe_str(lot_number).strip()
    if not house or not lot: return ""
    try: lot_int = int(lot) 
    except: lot_int = None

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

    def embed_image(self, img: Image.Image) -> np.ndarray:
        # --- CORRECTED FOR CLIP ---
        # 1. Preprocess and add batch dim
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 2. Use specific CLIP encode method (MobileNet used self.model(x))
            embedding = self.model.encode_image(img_tensor)
            
            # 3. L2 Normalize
            embedding = embedding / (embedding.norm(dim=-1, keepdim=True) + 1e-12)

            
        return embedding.cpu().numpy().flatten()

    def search(self, query_emb: np.ndarray, k: int = 10, period_filter: str = None, denom_filter: str = None) -> list[dict]:
        # Ensure query is normalized
        query_norm = np.linalg.norm(query_emb)
        if query_norm > 1e-12:
            query_emb = query_emb / query_norm
            
        # Matrix Multiply
        similarities = self.features @ query_emb
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in sorted_indices:
            row = self.metadata.iloc[idx]
            
            # Filters
            if period_filter and period_filter.lower() != "all":
                if period_filter.lower() not in str(row.get("parser_period", row.get("period", ""))).lower(): continue
            if denom_filter and denom_filter.lower() != "all":
                if denom_filter.lower() not in str(row.get("denomination", "")).lower(): continue
            
            results.append({
                "id": int(row.get("id", idx)),
                "image_path": _safe_str(row.get("image_path", "")),
                "period": _safe_str(row.get("parser_period", row.get("period", "unknown"))),
                "denomination": _safe_str(row.get("denomination", "unknown")),
                "cluster": _safe_str(row.get("visual_cluster", row.get("cluster_id", ""))),
                "similarity": float(similarities[idx]),
                "similarity_pct": f"{similarities[idx] * 100:.1f}%",
                "cosine": f"{similarities[idx]:.3f}",
                "lot_title": _safe_str(row.get("lot_title", row.get("title", ""))),
                "lot_url": _safe_str(row.get("lot_url", row.get("url", ""))) or build_lot_url(
                    row.get("auction_house", ""), row.get("sale_id", ""), row.get("lot_number", "")
                ),
            })
            if len(results) >= k: break
        return results

# =============================================================================
# Loader Logic (Hardened & Auto-Resolution)
# =============================================================================

def load_search_engine() -> SearchEngine:
    print(f"🔄 Loading CLIP model {CONFIG['clip_model']}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CONFIG["clip_model"], pretrained=CONFIG["clip_pretrained"], device=CONFIG["device"]
    )
    model.eval()
    
    print(f"🔄 Loading data from {CONFIG['features_path']}...")
    
    # 1. Hard Fail: File Existence Checks
    if not os.path.exists(CONFIG["features_path"]):
        raise FileNotFoundError(
            f"❌ Missing features file: {CONFIG['features_path']}. "
            "You must generate 512-dim CLIP features first."
        )
        
    if not os.path.exists(CONFIG["metadata_path"]):
        raise FileNotFoundError(
            f"❌ Missing metadata file: {CONFIG['metadata_path']}. "
            "Cannot map features to coin info."
        )

    # Load data
    features = np.load(CONFIG["features_path"])
    metadata = pd.read_csv(CONFIG["metadata_path"]).reset_index(drop=True)

    # 2. Hard Fail: Dimension Mismatch (Auto-Probed)
    feat_dim = int(features.shape[1])
    
    # Probe model output dimension using the actual preprocessing pipeline
    # This automatically handles 224px vs 336px vs other input sizes
    dummy_img = Image.new("RGB", (224, 224))  # Size doesn't matter, preprocess resizes it
    dummy_tensor = preprocess(dummy_img).unsqueeze(0).to(CONFIG["device"])
    
    with torch.no_grad():
        out = model.encode_image(dummy_tensor)
    model_dim = int(out.shape[-1])

    if feat_dim != model_dim:
        raise ValueError(
            f"❌ Embedding Dimension Mismatch!\n"
            f"   File '{CONFIG['features_path']}' has {feat_dim} dimensions.\n"
            f"   Loaded CLIP model outputs {model_dim} dimensions.\n"
            f"   FIX: Regenerate your features.npy using the {CONFIG['clip_model']} model."
        )

    # 3. Hard Fail: Row Alignment
    if len(features) != len(metadata):
        raise ValueError(
            f"❌ Data Alignment Error:\n"
            f"   Features file has {len(features)} rows.\n"
            f"   Metadata CSV has {len(metadata)} rows.\n"
            f"   These files must be from the same run to ensure IDs match."
        )

    # 4. Normalize (L2) for Cosine Similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.where(norms > 0, norms, 1)
    
    print(f"✅ Ready: {len(features)} coins indexed. Dimensions: {feat_dim}")
    return SearchEngine(features, metadata, model, preprocess, CONFIG["device"])
# =============================================================================
# FastAPI Routes (Keep your existing Routes below this line)
# =============================================================================

app = FastAPI(title="Trivalaya", version="0.5.0")
engine: SearchEngine = None

# --- NEW: Global Lock for 1 vCPU stability ---
search_lock = threading.Lock()

@contextmanager
def acquire_or_429(lock: threading.Lock):
    """
    Non-blocking context manager. 
    Raises 429 immediately if the lock is held by another request.
    """
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Server is busy processing another image.")
    try:
        yield
    finally:
        lock.release()
# ---------------------------------------------

@app.on_event("startup")
async def startup():
    global engine
    engine = load_search_engine()

@app.get("/", response_class=HTMLResponse)
async def home():
    # ... (Keep your existing home function code here) ...
    # (Copied from your original script for reference)
    periods = sorted(engine.metadata.get("parser_period", engine.metadata.get("period", pd.Series())).dropna().unique())
    p_opts = '<option value="all">All Periods</option>' + "".join(
        f'<option value="{html_module.escape(str(p))}">{html_module.escape(str(p))}</option>' for p in periods if p
    )
    denoms = sorted(engine.metadata.get("denomination", pd.Series()).dropna().unique())
    d_opts = '<option value="all">All Denominations</option>' + "".join(
        f'<option value="{html_module.escape(str(d))}">{html_module.escape(str(d))}</option>' for d in denoms if d
    )
    return HTML_TEMPLATE.replace("{{PERIOD_OPTIONS}}", p_opts).replace("{{DENOM_OPTIONS}}", d_opts)


# FastAPI Routes
@app.post("/search")
def search_endpoint(
    file: UploadFile = File(...),
    k: int = Query(10, ge=1, le=50),
    period: Optional[str] = None,
    denomination: Optional[str] = None,
    split: bool = Query(True, description="Attempt Obv/Rev split"),
):
    # This block ensures only 1 request processes at a time on your 1 vCPU
    with acquire_or_429(search_lock):
        
        # A. Validation
        if not (file.content_type or "").startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
        # B. Synchronous Read 
        contents = file.file.read()
        
        # Guard against empty/failed uploads
        if not contents:
            raise HTTPException(status_code=400, detail="Empty upload.")

        # C. Helper for running the search (CPU bound)
        def run_query(pil_img):
            emb = engine.embed_image(pil_img)
            return engine.search(emb, k=k, period_filter=period, denom_filter=denomination)

        split_result = {}
        
        # D. Split Heuristic (CPU bound - OpenCV)
        attempt_split = False
        if split and TWO_COIN_AVAILABLE:
            try:
                bgr = _bytes_to_bgr(contents)
                h, w = bgr.shape[:2]
                if w / h > 1.2: 
                    attempt_split = True
                else:
                    split_result = {"split_status": "skipped", "reason": "aspect_ratio_too_tall"}
            except HTTPException as he:
                raise he
            except Exception:
                attempt_split = False 

        if attempt_split:
            try:
                gray, binary = _prep_gray_and_binary(bgr)
                
                # Fast blob check
                bb_fast = _candidate_bbox_from_binary(binary)
                candidate_bbox = (0, 0, w, h)

                if bb_fast:
                    x, y, bw, bh = bb_fast
                    blob_ar = (bw / bh) if bh else 0.0
                    blob_area_ratio = (bw * bh) / float(h * w)
                    
                    if blob_ar < 1.15 or blob_area_ratio < 0.06:
                        split_result = {"split_status": "skipped", "reason": "blob_not_pair_like"}
                    else:
                        candidate_bbox = bb_fast
                        resolver = TwoCoinResolver()
                        try:
                            r = resolver.resolve(bgr, binary, gray, candidate_bbox)
                        except TypeError:
                            r = resolver.resolve(bgr, binary, gray)

                        if r.get("status") == "split" and len(r.get("coins", [])) == 2:
                            coins_out = []
                            for coin in r["coins"]:
                                crop_pil = Image.fromarray(cv2.cvtColor(coin["crop"], cv2.COLOR_BGR2RGB))
                                coins_out.append({
                                    "side": coin.get("side"),
                                    "bbox": coin.get("bbox"),
                                    "center": coin.get("center"),
                                    "radius": coin.get("radius"),
                                    "results": run_query(crop_pil),
                                })

                            return _json_safe({
                                "split_status": "split",
                                "split_method": r.get("method"),
                                "candidate_bbox": list(candidate_bbox),
                                "k": k,
                                "filters": {"period": period, "denomination": denomination},
                                "coins": coins_out,
                            })
                            
                        split_result = {"split_status": r.get("status", "failed"), "debug_method": r.get("method")}

            except Exception as e:
                print(f"Split failed: {e}")
                split_result = {"split_status": "error", "error": str(e)}

        # E. Fallback: Single Image Search
        try:
            pil = Image.open(BytesIO(contents)).convert("RGB")
            pil = ImageOps.exif_transpose(pil)
        except Exception:
            raise HTTPException(400, "Invalid image file (decoding failed).")

        return _json_safe({
            "split_status": "single_image",
            "debug": split_result,
            "k": k,
            "filters": {"period": period, "denomination": denomination},
            "results": run_query(pil),
        })

@app.get("/coin/{path:path}")
async def serve_image(path: str):
    # 1. Safer Path Resolution (Project Root Relative)
    if ".." in path or path.startswith("/"): raise HTTPException(400, "Bad path")
    
    # Try to resolve the path relative to the project root
    # This handles "trivalaya_data/img.jpg" correctly without double-nesting
    candidate = (project_root / path).resolve()
    
    # Security: Ensure the resolved path lies within one of the allowed data folders
    is_safe = False
    for allowed in CONFIG["allowed_dirs"]:
        allowed_root = (project_root / allowed).resolve()
        try:
            candidate.relative_to(allowed_root)
            is_safe = True
            break
        except ValueError:
            continue
            
    if is_safe and candidate.exists() and candidate.is_file():
        if candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            return FileResponse(candidate)
                
    raise HTTPException(404, "Image not found")

# =============================================================================
# Frontend
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
        
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1.5rem; }
        .card { background: var(--card); border: 1px solid #333; position: relative; transition: transform 0.2s; }
        .card:hover { transform: translateY(-3px); border-color: var(--accent); }
        .card img { width: 100%; aspect-ratio: 1; object-fit: contain; background: #000; }
        .info { padding: 1rem; }
        .score { color: var(--accent); font-family: monospace; font-size: 1.2rem; }
        .meta { font-size: 0.85rem; color: #888; margin-top: 0.5rem; }
        .badge { position: absolute; top: 0; right: 0; background: var(--accent); color: black; font-size: 0.7rem; padding: 2px 6px; font-weight: bold; }
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
                statusText = `Split Successful (${data.split_method}).`;
                data.coins.forEach(c => {
                    c.results.forEach(r => {
                        r._side = c.side; 
                        items.push(r);
                    });
                });
            } else {
                statusText = `Found ${data.results.length} matches (Single Image).`;
                // Show debug info if split was attempted but skipped/failed
                if (data.debug && data.debug.split_status) {
                    const r = data.debug.reason || data.debug.split_status;
                    statusText += ` [Split Skipped: ${r}]`;
                }
                items = data.results;
            }
            document.getElementById('status').innerText = statusText;

            items.forEach(r => {
                const div = document.createElement('div');
                div.className = 'card';
                
                // Safe path encoding for images with spaces/symbols
                const safePath = r.image_path.split('/').map(encodeURIComponent).join('/');
                const imgUrl = r.image_path.startsWith('http') ? r.image_path : `/coin/${safePath}`;
                
                const badge = r._side ? `<div class="badge">${r._side.toUpperCase()}</div>` : '';
                
                div.innerHTML = `
                    ${badge}
                    <img src="${imgUrl}" loading="lazy">
                    <div class="info">
                        <div class="score">${r.cosine}</div>
                        <div class="meta">${r.period} &middot; ${r.denomination}</div>
                        <div style="font-size:0.9rem; margin-top:0.5rem; font-weight:bold;">${r.lot_title || 'Untitled'}</div>
                        <div style="margin-top:0.5rem;"><a href="${r.lot_url}" target="_blank" style="color:#c9a227">View Lot</a></div>
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