#!/usr/bin/env python3
"""
cluster_coins.py - Visual clustering of coin images using CLIP embeddings.

Usage:
    python cluster_coins.py
    python cluster_coins.py --input_dir /path/to/images
    python cluster_coins.py --embeddings trivalaya_embeddings_pairs.npy --embeddings-meta trivalaya_embeddings_pairs_meta.json --skip-umap
    python cluster_coins.py --drill-down 25 --output cluster_output
"""

import os
import argparse
import json
import warnings
from pathlib import Path
from collections import defaultdict, Counter
from io import BytesIO
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import boto3
from urllib.parse import quote

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DB_HOST = os.getenv("TRIVALAYA_DB_HOST", "127.0.0.1")
DB_USER = os.getenv("TRIVALAYA_DB_USER", "auction_user")
DB_PASSWORD = os.getenv("TRIVALAYA_DB_PASSWORD", "Veritas@2024")
DB_NAME = os.getenv("TRIVALAYA_DB_NAME", "auction_data")
BATCH_SIZE = 32

_S3_CLIENT = None


# =============================================================================
# DATABASE & DATA LOADING
# =============================================================================

def get_db_connection():
    """Create database connection."""
    import mysql.connector
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )


def load_dataset_from_db(
    sample_size: Optional[int] = None,
    mode: str = "coin"
) -> List[Dict[str, Any]]:
    """
    Load dataset records from database.

    Args:
        sample_size: Optional limit on number of records.
        mode: "coin" for paired ml_coin_dataset, "legacy" for single-side ml_dataset.
    """
    print(f"📊 Loading dataset from database (mode={mode})...")
    records: List[Dict[str, Any]] = []
    try:
        with get_db_connection() as conn:
            with conn.cursor(dictionary=True) as cursor:
                if mode == "coin":
                    query = """
                        SELECT coin_id AS id, obv_path, rev_path,
                               has_obv, has_rev, period,
                               label_confidence, raw_label
                        FROM ml_coin_dataset
                        WHERE is_duplicate = 0
                          AND COALESCE(has_obv, 1) = 1
                          AND COALESCE(has_rev, 1) = 1
                          AND obv_path IS NOT NULL AND obv_path != ''
                          AND rev_path IS NOT NULL AND rev_path != ''
                          AND period IS NOT NULL AND period != ''
                    """
                else:
                    query = """
                        SELECT m.id, m.image_path, m.period,
                               m.label_confidence, m.raw_label
                        FROM ml_dataset m
                        WHERE m.image_path IS NOT NULL
                          AND m.image_path != ''
                          AND m.is_active = 1
                    """
                if sample_size:
                    query += f" ORDER BY RAND() LIMIT {int(sample_size)}"
                cursor.execute(query)
                records = cursor.fetchall()
    except Exception as e:
        print(f"❌ DB Error: {e}")

    print(f"  Loaded {len(records)} active records")
    return records


def load_dataset_from_directory(directory: str, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load images from folder and try to enrich with DB metadata using coin id prefix in filename.
    """
    print(f"📂 Scanning directory: {directory} ...")
    directory_path = Path(directory)
    if not directory_path.exists():
        print("❌ Directory does not exist.")
        return []

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    files = [f for f in directory_path.rglob("*") if f.suffix.lower() in image_extensions]

    if sample_size:
        import random
        random.shuffle(files)
        files = files[:sample_size]

    file_map: Dict[int, str] = {}
    fallback_records: List[Dict[str, Any]] = []
    fallback_id = -1  # negative IDs to avoid collisions with DB IDs

    for f in files:
        stem = f.name.split("_")[0]
        try:
            coin_id = int(stem)
            file_map[coin_id] = str(f.resolve())
        except ValueError:
            fallback_records.append({
                "id": fallback_id,
                "image_path": str(f.resolve()),
                "period": "unknown",
                "label_confidence": None,
                "raw_label": None,
            })
            fallback_id -= 1

    if not file_map:
        return fallback_records

    records: List[Dict[str, Any]] = []
    try:
        with get_db_connection() as conn:
            with conn.cursor(dictionary=True) as cursor:
                all_ids = list(file_map.keys())
                chunk = 1000
                for i in range(0, len(all_ids), chunk):
                    batch_ids = all_ids[i:i + chunk]
                    id_list = ",".join(map(str, batch_ids))
                    query = f"SELECT m.id, m.period, m.label_confidence, m.raw_label FROM ml_dataset m WHERE m.id IN ({id_list})"
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    for row in rows:
                        row["image_path"] = file_map[row["id"]]
                        records.append(row)
    except Exception:
        # If DB lookup fails, still proceed with path-only records
        records = [{
            "id": cid,
            "image_path": path,
            "period": "unknown",
            "label_confidence": None,
            "raw_label": None
        } for cid, path in file_map.items()]

    records.extend(fallback_records)
    print(f"  Loaded {len(records)} records from directory mode")
    return records


# =============================================================================
# PATH / SPACES HELPERS
# =============================================================================
def img_src(raw_path: str) -> str:
    """
    Build browser-safe URL for your image proxy endpoint.
    Expects app route: /img?path=<raw_path>
    """
    return f"/img?path={quote(str(raw_path), safe='/')}"

def _path_keys(p: str) -> List[str]:
    """Generate equivalent path keys for matching across rel/abs differences."""
    s = str(p).replace("\\", "/").strip()
    keys = {s, s.lstrip("./"), s.lstrip("/")}
    try:
        keys.add(str(Path(s).resolve()).replace("\\", "/"))
    except Exception:
        pass
    if "trivalaya_data/" in s:
        tail = s.split("trivalaya_data/", 1)[1]
        keys.add(tail)
        keys.add("trivalaya_data/" + tail)
    return [k for k in keys if k]


def resolve_image_path(db_path: str, base_path: str = "") -> Optional[Path]:
    if not db_path:
        return None
    p = Path(db_path)
    candidates = [p]
    if not p.is_absolute():
        if base_path:
            candidates.append(Path(base_path) / p)
        candidates.extend([Path(".") / p, Path("..") / p, Path.home() / p, Path("/root") / p])

    for c in candidates:
        if c.exists():
            return c
    return None


def _is_spaces_key(p: str) -> bool:
    if not p:
        return False
    s = p.strip().lstrip("/")
    return s.startswith("processed/") or s.startswith("raw/")


def _validate_spaces_env_if_needed() -> None:
    # Only validate if Spaces might be used
    required = ["SPACES_BUCKET", "SPACES_ENDPOINT", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing required Spaces env vars: {', '.join(missing)}")


def _get_s3():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        bucket = os.getenv("SPACES_BUCKET")
        if not bucket:
            raise RuntimeError("SPACES_BUCKET is not set")
        _S3_CLIENT = boto3.client(
            "s3",
            endpoint_url=os.getenv("SPACES_ENDPOINT"),
            region_name=os.getenv("SPACES_REGION", "sfo3"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    return _S3_CLIENT


def open_image_local_or_spaces(path: str, base_path: Optional[str] = None) -> Image.Image:
    """
    Try local first (resolved with base_path), then Spaces key.
    Returns an RGB PIL image.
    """
    local_path = path
    if base_path and path and not os.path.isabs(path):
        local_path = os.path.join(base_path, path)

    if local_path and os.path.exists(local_path):
        with Image.open(local_path) as im:
            return im.convert("RGB")

    if _is_spaces_key(path):
        s3 = _get_s3()
        obj = s3.get_object(Bucket=os.getenv("SPACES_BUCKET"), Key=path.lstrip("/"))
        with Image.open(BytesIO(obj["Body"].read())) as im:
            return im.convert("RGB")

    raise FileNotFoundError(f"Not found locally or on Spaces: {path}")


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def load_clip_model(use_gpu: bool = False):
    print("🧠 Loading CLIP model...")
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        print("  Loaded open_clip ViT-B-32")
        return model.to(device).eval(), preprocess, device, "open_clip"
    except ImportError:
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("  Loaded OpenAI CLIP ViT-B/32")
        return model.eval(), preprocess, device, "clip"


def process_batch(images: List[torch.Tensor], model, device: str, model_type: str) -> List[np.ndarray]:
    batch = torch.stack(images).to(device)
    with torch.no_grad():
        # same call path for both open_clip and clip
        features = model.encode_image(batch)
    features = features.detach().cpu().numpy().astype(np.float32)
    return [f / (np.linalg.norm(f) + 1e-12) for f in features]


def extract_features(
    records: List[Dict[str, Any]],
    model,
    preprocess,
    device: str,
    model_type: str,
    base_path: str = ""
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    print(f"\n🔍 Extracting features from {len(records)} images...")
    features: List[np.ndarray] = []
    valid_records: List[Dict[str, Any]] = []
    stats = {"local_missing": 0, "spaces_missing": 0, "decode_error": 0}

    batch_images: List[torch.Tensor] = []
    batch_records: List[Dict[str, Any]] = []

    for record in tqdm(records, desc="Processing"):
        p = str(record.get("image_path", ""))
        try:
            img = open_image_local_or_spaces(p, base_path=base_path)
            batch_images.append(preprocess(img))
            batch_records.append(record)

            if len(batch_images) >= BATCH_SIZE:
                features.extend(process_batch(batch_images, model, device, model_type))
                valid_records.extend(batch_records)
                batch_images, batch_records = [], []
        except FileNotFoundError:
            if _is_spaces_key(p):
                stats["spaces_missing"] += 1
            else:
                stats["local_missing"] += 1
        except Exception:
            stats["decode_error"] += 1

    if batch_images:
        features.extend(process_batch(batch_images, model, device, model_type))
        valid_records.extend(batch_records)

    print(f"  Extracted: {len(features)} | Errors: {stats}")
    return np.asarray(features, dtype=np.float32), valid_records


def extract_features_paired(
    records: List[Dict[str, Any]],
    model,
    preprocess,
    device: str,
    model_type: str,
    base_path: str = ""
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Extract concatenated obverse+reverse CLIP features per coin.

    Each coin produces a single 1024-d vector (512-d obv ∥ 512-d rev),
    normalized per-side then normalized again after concatenation to
    preserve cosine geometry for UMAP/HDBSCAN.

    Uses batched forward passes for efficiency.
    """
    print(f"\n🔍 Extracting paired features from {len(records)} coins...")
    features: List[np.ndarray] = []
    valid_records: List[Dict[str, Any]] = []
    stats = {"missing": 0, "decode_error": 0}

    obv_batch: List[torch.Tensor] = []
    rev_batch: List[torch.Tensor] = []
    batch_records: List[Dict[str, Any]] = []

    def _flush_batch():
        if not obv_batch:
            return
        obv_stack = torch.stack(obv_batch).to(device)
        rev_stack = torch.stack(rev_batch).to(device)
        with torch.no_grad():
            obv_feats = model.encode_image(obv_stack).cpu().numpy().astype(np.float32)
            rev_feats = model.encode_image(rev_stack).cpu().numpy().astype(np.float32)
        # Normalize each side independently
        obv_feats = obv_feats / (np.linalg.norm(obv_feats, axis=1, keepdims=True) + 1e-12)
        rev_feats = rev_feats / (np.linalg.norm(rev_feats, axis=1, keepdims=True) + 1e-12)
        # Concatenate → 1024-d, then normalize the combined vector
        combined = np.concatenate([obv_feats, rev_feats], axis=1)
        combined = combined / (np.linalg.norm(combined, axis=1, keepdims=True) + 1e-12)
        features.extend(combined)
        valid_records.extend(batch_records)
        obv_batch.clear()
        rev_batch.clear()
        batch_records.clear()

    for record in tqdm(records, desc="Processing coins"):
        obv_path = str(record.get("obv_path", ""))
        rev_path = str(record.get("rev_path", ""))
        try:
            obv_img = open_image_local_or_spaces(obv_path, base_path=base_path)
            rev_img = open_image_local_or_spaces(rev_path, base_path=base_path)
            obv_batch.append(preprocess(obv_img))
            rev_batch.append(preprocess(rev_img))
            batch_records.append(record)

            if len(obv_batch) >= BATCH_SIZE:
                _flush_batch()
        except FileNotFoundError:
            stats["missing"] += 1
        except Exception:
            stats["decode_error"] += 1

    _flush_batch()

    print(f"  Extracted: {len(features)} paired features (1024-d) | Errors: {stats}")
    return np.asarray(features, dtype=np.float32), valid_records


# =============================================================================
# PRECOMPUTED EMBEDDINGS
# =============================================================================

def load_precomputed_features(
    records: List[Dict[str, Any]],
    embeddings_path: str,
    meta_path: str,
    base_path: str = "",
    mode: str = "coin"
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    print(f"📦 Loading precomputed embeddings from {embeddings_path}...")
    vecs = np.load(embeddings_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Single-side meta → index by path
    if "paths" in meta:
        path_list = meta["paths"]
        if len(path_list) != vecs.shape[0]:
            raise ValueError(f"Mismatch: {vecs.shape[0]} vectors vs {len(path_list)} paths")

        index: Dict[str, int] = {}
        for i, p in enumerate(path_list):
            for k in _path_keys(str(p)):
                index[k] = i

        features: List[np.ndarray] = []
        valid_records: List[Dict[str, Any]] = []
        missing = 0

        for r in records:
            raw_path = str(r.get("image_path", "")).strip()
            candidates = _path_keys(raw_path)
            resolved = resolve_image_path(raw_path, base_path=base_path)
            if resolved is not None:
                candidates += _path_keys(str(resolved))
            hit = next((index[k] for k in candidates if k in index), None)
            if hit is not None:
                features.append(vecs[hit])
                valid_records.append(r)
            else:
                missing += 1

    # Pair meta → index by (obv_key, rev_key) tuple to avoid mismatch
    elif "obverse_paths" in meta:
        obv_paths = meta["obverse_paths"]
        rev_paths = meta.get("reverse_paths", [])
        if len(obv_paths) != vecs.shape[0]:
            raise ValueError(f"Mismatch: {vecs.shape[0]} vectors vs {len(obv_paths)} obverse_paths")
        if len(rev_paths) != len(obv_paths):
            raise ValueError(f"Mismatch: {len(obv_paths)} obverse vs {len(rev_paths)} reverse paths")

        # Build index keyed by canonical (obv, rev) tuple
        pair_index: Dict[Tuple[str, str], int] = {}
        for i in range(len(obv_paths)):
            obv_keys = _path_keys(str(obv_paths[i]))
            rev_keys = _path_keys(str(rev_paths[i]))
            for ok in obv_keys:
                for rk in rev_keys:
                    pair_index[(ok, rk)] = i
        print(f"  Detected pair metadata; indexed {len(obv_paths)} coin pairs")

        features = []
        valid_records = []
        missing = 0

        for r in records:
            if mode == "coin":
                raw_obv = str(r.get("obv_path", ""))
                raw_rev = str(r.get("rev_path", ""))
            else:
                # Legacy single-side: can't do pair matching
                raw_obv = str(r.get("image_path", ""))
                raw_rev = ""

            obv_candidates = _path_keys(raw_obv)
            rev_candidates = _path_keys(raw_rev) if raw_rev else [""]

            resolved_obv = resolve_image_path(raw_obv, base_path=base_path)
            resolved_rev = resolve_image_path(raw_rev, base_path=base_path) if raw_rev else None
            if resolved_obv:
                obv_candidates += _path_keys(str(resolved_obv))
            if resolved_rev:
                rev_candidates += _path_keys(str(resolved_rev))

            hit = None
            for ok in obv_candidates:
                for rk in rev_candidates:
                    if (ok, rk) in pair_index:
                        hit = pair_index[(ok, rk)]
                        break
                if hit is not None:
                    break

            if hit is not None:
                features.append(vecs[hit])
                valid_records.append(r)
            else:
                missing += 1

    else:
        raise ValueError("Invalid metadata format. Expected 'paths' or 'obverse_paths'.")

    if len(features) == 0:
        raise RuntimeError("No records matched embeddings metadata paths.")

    feats = np.asarray(features, dtype=np.float32)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)

    print(f"  ✅ Aligned {len(valid_records)}/{len(records)} records (Missing: {missing})")
    return feats, valid_records


# =============================================================================
# CLUSTERING / ANALYSIS
# =============================================================================

def cluster_features(
    features: np.ndarray,
    min_cluster_size: int = 20,
    min_samples: int = 10,
    skip_umap: bool = False,
    pca_dims: Optional[int] = None,
):
    """Cluster features using HDBSCAN, optionally on raw embeddings or PCA/UMAP reduced."""
    print(f"\n🎯 Clustering {len(features)} feature vectors...")
    import umap
    import hdbscan

    if pca_dims:
        from sklearn.decomposition import PCA
        print(f"  Running PCA ({features.shape[1]}-d → {pca_dims}-d) for clustering...")
        pca = PCA(n_components=pca_dims, random_state=42)
        clustering_space = pca.fit_transform(features).astype(np.float32)
        explained = float(pca.explained_variance_ratio_.sum()) * 100.0
        print(f"  PCA explained variance: {explained:.1f}%")
    elif skip_umap:
        print(f"  Clustering on raw {features.shape[1]}-d embeddings (skip-umap mode)...")
        clustering_space = features.astype(np.float32)
    else:
        print("  Running UMAP (15-d for clustering)...")
        reducer = umap.UMAP(
            n_components=15, n_neighbors=30, min_dist=0.0, metric="cosine", random_state=42
        )
        clustering_space = reducer.fit_transform(features).astype(np.float32)

    print("  Creating 2D projection for visualization...")
    reducer_2d = umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42
    )
    embedding_2d = reducer_2d.fit_transform(features).astype(np.float32)

    print(f"  Running HDBSCAN (min_cluster={min_cluster_size}, min_samples={min_samples})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    )
    cluster_labels = clusterer.fit_predict(clustering_space)
    confidences = getattr(clusterer, "probabilities_", np.ones(len(cluster_labels), dtype=np.float32))

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int(np.sum(cluster_labels == -1))
    noise_pct = 100.0 * n_noise / max(1, len(cluster_labels))
    print(f"  Found {n_clusters} clusters, {n_noise} noise points ({noise_pct:.1f}%)")

    return cluster_labels, confidences, embedding_2d


def analyze_clusters(records, cluster_labels, confidences):
    """Analyze cluster composition."""
    print("\n📈 Analyzing cluster composition...")
    cluster_data = defaultdict(list)

    for record, cluster, conf in zip(records, cluster_labels, confidences):
        cluster_data[int(cluster)].append({
            "id": record.get("id"),
            "path": record.get("image_path") or record.get("obv_path") or record.get("rev_path"),
            "period": record.get("period") or "unknown",
            "confidence": float(conf),
        })

    summary = []
    for cluster_id in sorted(cluster_data.keys()):
        items = cluster_data[cluster_id]
        periods = Counter(item["period"] for item in items)
        dominant = periods.most_common(1)[0] if periods else ("unknown", 0)
        purity = (dominant[1] / len(items)) if items else 0.0

        summary.append({
            "cluster": f"cluster_{cluster_id}" if cluster_id >= 0 else "noise",
            "cluster_id": int(cluster_id),
            "size": int(len(items)),
            "dominant_period": dominant[0],
            "purity": float(purity),
            "period_breakdown": dict(periods),
            "avg_confidence": float(np.mean([item["confidence"] for item in items])) if items else 0.0,
        })

    summary.sort(key=lambda x: x["size"], reverse=True)
    return cluster_data, summary


def save_results(records, cluster_labels, confidences, embedding_2d, summary, output_dir, features=None):
    """Save clustering outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving results to {output_dir}/")

    df = pd.DataFrame([{
        "id": r.get("id"),
        "image_path": r.get("image_path", ""),
        "obv_path": r.get("obv_path", ""),
        "rev_path": r.get("rev_path", ""),
        "parser_period": r.get("period") or "unknown",
        "visual_cluster": f"cluster_{int(cl)}" if int(cl) >= 0 else "noise",
        "cluster_id": int(cl),
        "cluster_confidence": float(conf),
        "umap_x": float(embedding_2d[i, 0]),
        "umap_y": float(embedding_2d[i, 1]),
    } for i, (r, cl, conf) in enumerate(zip(records, cluster_labels, confidences))])

    df.to_csv(output_dir / "cluster_results.csv", index=False)
    print(f"  ✓ cluster_results.csv ({len(df)} rows)")

    if features is not None:
        np.save(output_dir / "features.npy", features)
        print("  ✓ features.npy (saved for drill-down)")

    with open(output_dir / "cluster_summary.txt", "w") as f:
        n_clusters = len([s for s in summary if s["cluster_id"] >= 0])
        f.write(f"Clusters found: {n_clusters}\n")
        for info in summary:
            f.write(f"\n{info['cluster'].upper()} (n={info['size']})\n")
            f.write(f"  Dominant: {info['dominant_period']} ({info['purity']*100:.1f}%)\n")
            f.write(f"  Breakdown: {info['period_breakdown']}\n")
    print("  ✓ cluster_summary.txt")

    with open(output_dir / "cluster_data.json", "w") as f:
        json.dump({"clusters": summary}, f, indent=2, default=str)
    print("  ✓ cluster_data.json")

    return df


def generate_html_visualization(df, records, cluster_labels, summary, output_dir):
    """Generate HTML grid visualization using /img proxy paths."""
    output_dir = Path(output_dir)
    print("  Generating HTML visualization...")

    clusters_html = []
    for info in summary:
        cluster_id = info["cluster_id"]
        cluster_items = [(r, i) for i, (r, cl) in enumerate(zip(records, cluster_labels)) if int(cl) == int(cluster_id)]

        import random
        sample = random.sample(cluster_items, min(20, len(cluster_items))) if cluster_items else []

        images_html = []
        for record, _ in sample:
            period = record.get("period") or "unknown"
            obv_path = record.get("obv_path")
            rev_path = record.get("rev_path")

            # Paired mode: obv + rev
            if obv_path and rev_path:
                obv_url = img_src(obv_path)
                rev_url = img_src(rev_path)
                images_html.append(f"""
                <div class="coin-card paired clustered">
                    <div class="coin-pair">
                        <img src="{obv_url}" alt="obv" loading="lazy" title="Obverse">
                        <img src="{rev_url}" alt="rev" loading="lazy" title="Reverse">
                    </div>
                    <div class="coin-info"><span class="period">{period}</span></div>
                </div>
                """)
                continue

            # Fallback: single image
            raw = record.get("image_path") or obv_path or rev_path or ""
            if raw:
                url = img_src(raw)
                images_html.append(f"""
                <div class="coin-card clustered">
                    <img src="{url}" alt="coin" loading="lazy">
                    <div class="coin-info"><span class="period">{period}</span></div>
                </div>
                """)

        clusters_html.append(f"""
        <div class="cluster" id="cluster-{cluster_id}">
            <div class="cluster-header">
                <h2>{info['cluster']} ({info['size']})</h2>
                <div class="purity">Dominant: {info['dominant_period']} ({info['purity']*100:.1f}%)</div>
            </div>
            <div class="coin-grid">{''.join(images_html)}</div>
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Trivalaya Visual Clusters</title>
    <style>
        body {{ background: #1a1a2e; color: #eee; font-family: sans-serif; padding: 20px; }}
        .cluster {{ background: #16213e; margin-bottom: 30px; border-radius: 8px; overflow: hidden; }}
        .cluster-header {{ padding: 15px; border-bottom: 1px solid #333; }}
        .coin-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; padding: 15px; }}
        .coin-card img {{ width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 4px; }}
        .coin-card.paired .coin-pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 3px; }}
        .coin-card.paired .coin-pair img {{ width: 100%; }}
        .coin-card.clustered {{ border: 2px solid transparent; }}
        h1 {{ color: #ffd700; margin-top: 0; }}
        h2 {{ color: #ffd700; margin: 0; }}
        .period {{ font-size: 12px; color: #ccc; }}
    </style>
</head>
<body>
    <h1>🪙 Trivalaya Visual Clusters</h1>
    {''.join(clusters_html)}
</body>
</html>"""

    with open(output_dir / "cluster_visualization.html", "w") as f:
        f.write(html)
    print("  ✓ cluster_visualization.html")




def drill_down_cluster(target_cluster_id, features, df, output_dir):
    """Isolate one cluster and break it into sub-clusters."""
    print(f"\n⛏️ DRILLING DOWN into Cluster {target_cluster_id}...")
    mask = df["cluster_id"] == target_cluster_id
    indices = df.index[mask].tolist()

    if not indices:
        print(f"❌ Cluster {target_cluster_id} not found or empty.")
        return

    sub_features = features[indices]
    sub_records = df.loc[mask].to_dict("records")
    for r in sub_records:
        if "parser_period" in r:
            r["period"] = r.pop("parser_period")

    print(f"   Analyzing {len(sub_features)} coins in sub-space...")

    import umap
    import hdbscan

    reducer = umap.UMAP(n_components=5, n_neighbors=100, min_dist=0.2, metric="cosine", random_state=42)
    sub_embedding = reducer.fit_transform(sub_features)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2, cluster_selection_method="leaf")
    sub_labels = clusterer.fit_predict(sub_embedding)

    sub_output = Path(output_dir) / f"subcluster_{target_cluster_id}"
    sub_output.mkdir(parents=True, exist_ok=True)

    viz_reducer = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.1, metric="cosine", random_state=42)
    sub_embedding_2d = viz_reducer.fit_transform(sub_features)
    sub_conf = np.ones(len(sub_labels), dtype=np.float32)

    _, sub_summary = analyze_clusters(sub_records, sub_labels, sub_conf)
    save_results(sub_records, sub_labels, sub_conf, sub_embedding_2d, sub_summary, sub_output, features=sub_features)
    generate_html_visualization(None, sub_records, sub_labels, sub_summary, sub_output)
    print(f"\n✅ Sub-clustering complete! See: {sub_output / 'cluster_visualization.html'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visual clustering of coin images")
    parser.add_argument("--sample", type=int, help="Sample N images")
    parser.add_argument("--min-cluster", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output", type=str, default="cluster_output")
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--drill-down", type=int)
    parser.add_argument("--base-path", type=str, default=os.getenv("TRIVALAYA_LOCAL_DATA", ""))
    parser.add_argument("--embeddings", type=str)
    parser.add_argument("--embeddings-meta", type=str)
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--pca", type=int)
    parser.add_argument(
        "--mode", choices=["coin", "legacy"], default="coin",
        help="'coin' reads paired obv/rev from ml_coin_dataset (default); "
             "'legacy' reads single-side from ml_dataset"
    )

    args = parser.parse_args()

    print("🪙 TRIVALAYA VISUAL CLUSTERING")
    print("=" * 50)
    print(f"  Mode: {args.mode}")

    # Drill-down mode
    if args.drill_down is not None:
        out = Path(args.output)
        f_path = out / "features.npy"
        csv_path = out / "cluster_results.csv"
        if not f_path.exists() or not csv_path.exists():
            print("❌ Missing features.npy or cluster_results.csv. Run a full cluster first.")
            return
        features = np.load(f_path)
        df = pd.read_csv(csv_path)
        drill_down_cluster(args.drill_down, features, df, args.output)
        return

    # Load records
    if args.input_dir:
        records = load_dataset_from_directory(args.input_dir, args.sample)
    else:
        records = load_dataset_from_db(args.sample, mode=args.mode)
    if not records:
        print("❌ No records found.")
        return

    # Features
    if args.embeddings:
        if not args.embeddings_meta:
            raise ValueError("--embeddings-meta is required when using --embeddings")
        features, valid_records = load_precomputed_features(
            records, args.embeddings, args.embeddings_meta,
            base_path=args.base_path, mode=args.mode
        )
    else:
        # If many image paths look like Spaces keys, validate env early
        sample_paths = []
        for r in records[:200]:
            sample_paths.append(str(r.get("obv_path", r.get("image_path", ""))))
        if any(_is_spaces_key(p) for p in sample_paths):
            _validate_spaces_env_if_needed()

        model, preprocess, device, mtype = load_clip_model(args.gpu)

        if args.mode == "coin":
            features, valid_records = extract_features_paired(
                records, model, preprocess, device, mtype, base_path=args.base_path
            )
        else:
            features, valid_records = extract_features(
                records, model, preprocess, device, mtype, base_path=args.base_path
            )

    if len(features) < 5:
        print("❌ Too few valid images.")
        return

    labels, conf, embed2d = cluster_features(
        features,
        min_cluster_size=args.min_cluster,
        min_samples=args.min_samples,
        skip_umap=args.skip_umap,
        pca_dims=args.pca
    )
    _, summary = analyze_clusters(valid_records, labels, conf)
    save_results(valid_records, labels, conf, embed2d, summary, args.output, features=features)
    generate_html_visualization(None, valid_records, labels, summary, args.output)

    print(f"\n✅ Done! Results in {args.output}/")


if __name__ == "__main__":
    main()
