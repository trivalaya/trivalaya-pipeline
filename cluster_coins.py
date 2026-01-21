#!/usr/bin/env python3
"""
cluster_coins.py - Visual clustering of coin images using CLIP embeddings.

Usage:
    python cluster_coins.py                    # Full run (DB)
    python cluster_coins.py --input_dir ...    # Run on a folder (Directory Mode)
    python cluster_coins.py --drill-down 25    # Split Cluster 25 (Sub-clustering)
    python cluster_coins.py --embeddings trivalaya_embeddings.npy --embeddings-meta trivalaya_embeddings_meta.json
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch

# --- CONFIGURATION ---
DB_HOST = os.getenv("TRIVALAYA_DB_HOST", "127.0.0.1")
DB_USER = os.getenv("TRIVALAYA_DB_USER", "auction_user")
DB_PASSWORD = os.getenv("TRIVALAYA_DB_PASSWORD", "Veritas@2024")
DB_NAME = os.getenv("TRIVALAYA_DB_NAME", "auction_data")

BATCH_SIZE = 32

def get_db_connection():
    """Create database connection."""
    import mysql.connector
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

def load_dataset_from_db(sample_size=None):
    """Load image paths and periods from ml_dataset."""
    print("📊 Loading dataset from database...")
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    query = """
        SELECT m.id, m.image_path, m.period, m.label_confidence, m.raw_label
        FROM ml_dataset m
        WHERE m.image_path IS NOT NULL 
        AND m.image_path != ''
        AND m.is_active = 1
    """
    if sample_size:
        query += f" ORDER BY RAND() LIMIT {sample_size}"
    
    cursor.execute(query)
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    
    print(f"  Loaded {len(records)} active records")
    return records

def load_dataset_from_directory(directory, sample_size=None):
    """Load images and recover metadata from DB using ID prefixes."""
    print(f"📂 Scanning directory: {directory}...")
    directory = Path(directory)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif'}
    files = [f for f in directory.rglob("*") if f.suffix.lower() in image_extensions]
    
    if sample_size:
        import random
        random.shuffle(files)
        files = files[:sample_size]

    # 1. Extract IDs from filenames
    file_map = {} 
    for f in files:
        try:
            coin_id = int(f.name.split('_')[0])
            file_map[coin_id] = str(f.absolute())
        except ValueError:
            continue

    if not file_map:
        print("  ❌ No valid IDs found in filenames. Returning basic records.")
        return [{'id': i, 'image_path': str(f), 'period': 'unknown'} for i, f in enumerate(files)]

    print(f"  Recovered IDs for {len(file_map)} images. Fetching metadata...")

    # 2. Bulk Fetch Metadata
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    all_ids = list(file_map.keys())
    records = []
    batch_size = 1000
    
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i + batch_size]
        id_list = ",".join(map(str, batch_ids))
        
        query = f"""
            SELECT m.id, m.period, m.label_confidence, m.raw_label
            FROM ml_dataset m
            WHERE m.id IN ({id_list})
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        for row in rows:
            row['image_path'] = file_map[row['id']]
            records.append(row)
            
    cursor.close()
    conn.close()
    print(f"  Merged metadata for {len(records)} records")
    return records

def resolve_image_path(db_path):
    """Find actual file location."""
    path_obj = Path(db_path)
    if path_obj.is_absolute() and path_obj.exists():
        return path_obj

    candidates = [
        path_obj,
        Path(".") / db_path,
        Path("..") / db_path,
        Path("/root") / db_path,
        Path.home() / db_path,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def load_clip_model(use_gpu=False):
    """Load CLIP model."""
    print("🧠 Loading CLIP model...")
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        model = model.to(device)
        model.eval()
        print("  Loaded open_clip ViT-B-32")
        return model, preprocess, device, "open_clip"
    except ImportError:
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        print("  Loaded OpenAI CLIP ViT-B/32")
        return model, preprocess, device, "clip"

def extract_features(records, model, preprocess, device, model_type):
    """Extract visual features."""
    print(f"\n🔍 Extracting features from {len(records)} images...")
    features = []
    valid_records = []
    failed = 0
    
    batch_images = []
    batch_records = []
    
    for record in tqdm(records, desc="Loading images"):
        path = resolve_image_path(record['image_path'])
        if not path:
            failed += 1
            continue
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = preprocess(img)
            batch_images.append(img_tensor)
            batch_records.append(record)
            
            if len(batch_images) >= BATCH_SIZE:
                batch_features = process_batch(batch_images, model, device, model_type)
                features.extend(batch_features)
                valid_records.extend(batch_records)
                batch_images = []
                batch_records = []
        except Exception:
            failed += 1
            continue
    
    if batch_images:
        batch_features = process_batch(batch_images, model, device, model_type)
        features.extend(batch_features)
        valid_records.extend(batch_records)
    
    print(f"  Extracted: {len(features)}, Failed: {failed}")
    return np.array(features), valid_records

def process_batch(images, model, device, model_type):
    batch = torch.stack(images).to(device)
    with torch.no_grad():
        if model_type == "open_clip":
            features = model.encode_image(batch)
        else:
            features = model.encode_image(batch)
    features = features.cpu().numpy()
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    return list(features)

# =============================================================================
# PRECOMPUTED EMBEDDINGS SUPPORT (v5.2)
# =============================================================================

def _path_keys(p: str) -> list:
    """Generate equivalent path keys for matching across rel/abs differences."""
    s = str(p).replace("\\", "/")
    keys = {s, s.lstrip("./")}
    try:
        keys.add(str(Path(p).resolve()).replace("\\", "/"))
    except Exception:
        pass
    if "trivalaya_data/" in s:
        keys.add(s.split("trivalaya_data/", 1)[1])
        keys.add("trivalaya_data/" + s.split("trivalaya_data/", 1)[1])
    return list(keys)

def load_precomputed_features(records, embeddings_path, meta_path):
    """
    Load precomputed embeddings (e.g., from v5.2 MobileNetV2) and align with records.
    
    Args:
        records: List of dicts with 'image_path' key
        embeddings_path: Path to .npy file with embeddings (N x D)
        meta_path: Path to JSON with 'paths' list aligning to embeddings
    
    Returns:
        features: numpy array of matched embeddings
        valid_records: list of records that matched
    """
    print(f"📦 Loading precomputed embeddings from {embeddings_path}...")
    
    vecs = np.load(embeddings_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    meta_paths = meta["paths"]
    if len(meta_paths) != vecs.shape[0]:
        raise ValueError(f"Mismatch: {vecs.shape[0]} vectors vs {len(meta_paths)} paths")

    print(f"  Loaded {vecs.shape[0]} embeddings (dim={vecs.shape[1]})")

    # Build path -> row index mapping
    index = {}
    for i, p in enumerate(meta_paths):
        for k in _path_keys(p):
            index[k] = i

    features = []
    valid_records = []
    missing = 0

    for r in records:
        p = r["image_path"]
        resolved = resolve_image_path(p)
        candidates = _path_keys(p)
        if resolved is not None:
            candidates += _path_keys(str(resolved))

        hit = None
        for k in candidates:
            if k in index:
                hit = index[k]
                break

        if hit is None:
            missing += 1
            continue

        features.append(vecs[hit])
        valid_records.append(r)

    if not features:
        raise RuntimeError("No records matched embeddings meta paths.")

    features = np.asarray(features, dtype=np.float32)

    # Ensure normalized for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    features = features / norms

    print(f"  ✅ Using precomputed embeddings: matched {len(valid_records)}/{len(records)} (missing {missing})")
    return features, valid_records

# =============================================================================
# CLUSTERING
# =============================================================================

def cluster_features(features, min_cluster_size=20, min_samples=10, skip_umap=False, pca_dims=None):
    """Cluster features using HDBSCAN, optionally on raw embeddings or PCA/UMAP reduced."""
    print(f"\n🎯 Clustering {len(features)} feature vectors...")
    import umap
    import hdbscan

    # --- Choose clustering space ---
    if pca_dims:
        from sklearn.decomposition import PCA
        print(f"  Running PCA ({features.shape[1]}-d → {pca_dims}-d) for clustering...")
        pca = PCA(n_components=pca_dims, random_state=42)
        clustering_space = pca.fit_transform(features).astype(np.float32)
        print(f"  PCA explained variance: {float(pca.explained_variance_ratio_.sum())*100:.1f}%")
    elif skip_umap:
        print(f"  Clustering on raw {features.shape[1]}-d embeddings (skip-umap mode)...")
        clustering_space = features.astype(np.float32)
    else:
        print("  Running UMAP (15-d for clustering)...")
        reducer = umap.UMAP(
            n_components=15, n_neighbors=30, min_dist=0.0,
            metric='cosine', random_state=42
        )
        clustering_space = reducer.fit_transform(features).astype(np.float32)

    # --- 2D projection for visualization ---
    print("  Creating 2D projection for visualization...")
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=30,   # 100 looks nicer but slower; 30 is good default
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    embedding_2d = reducer_2d.fit_transform(features).astype(np.float32)

    # --- HDBSCAN ---
    print(f"  Running HDBSCAN (min_cluster={min_cluster_size}, min_samples={min_samples})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        core_dist_n_jobs=-1  # use all CPU cores
    )
    cluster_labels = clusterer.fit_predict(clustering_space)

    # Cheap per-point confidence (avoid expensive membership_vector call)
    confidences = getattr(clusterer, "probabilities_", np.ones(len(cluster_labels), dtype=np.float32))

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int(np.sum(cluster_labels == -1))
    print(f"  Found {n_clusters} clusters, {n_noise} noise points ({100*n_noise/len(cluster_labels):.1f}%)")

    return cluster_labels, confidences, embedding_2d

def analyze_clusters(records, cluster_labels, confidences):
    """Analyze composition."""
    print("\n📈 Analyzing cluster composition...")
    cluster_data = defaultdict(list)
    
    for record, cluster, conf in zip(records, cluster_labels, confidences):
        cluster_data[cluster].append({
            'id': record['id'],
            'path': record['image_path'],
            'period': record['period'] or 'unknown',
            'confidence': conf
        })
    
    summary = []
    for cluster_id in sorted(cluster_data.keys()):
        items = cluster_data[cluster_id]
        periods = Counter(item['period'] for item in items)
        dominant = periods.most_common(1)[0] if periods else ('unknown', 0)
        purity = dominant[1] / len(items) if items else 0
        
        summary.append({
            'cluster': f"cluster_{cluster_id}" if cluster_id >= 0 else "noise",
            'cluster_id': cluster_id,
            'size': len(items),
            'dominant_period': dominant[0],
            'purity': purity,
            'period_breakdown': dict(periods),
            'avg_confidence': np.mean([item['confidence'] for item in items])
        })
    
    summary.sort(key=lambda x: x['size'], reverse=True)
    return cluster_data, summary

def suggest_cluster_names(summary):
    suggestions = {}
    for info in summary:
        if info['cluster_id'] < 0: continue
        dom = info['dominant_period']
        if info['purity'] > 0.8:
            suggestions[info['cluster_id']] = f"{dom}_pure"
        elif info['purity'] > 0.5:
            suggestions[info['cluster_id']] = f"{dom}_mixed"
        else:
            suggestions[info['cluster_id']] = "mixed_group"
    return suggestions

def save_results(records, cluster_labels, confidences, embedding_2d, summary, output_dir, features=None):
    """Save all results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 Saving results to {output_dir}/")
    
    # 1. Save CSV
    df = pd.DataFrame([{
        'id': r['id'],
        'image_path': r['image_path'],
        'parser_period': r['period'] or 'unknown',
        'visual_cluster': f"cluster_{cl}" if cl >= 0 else "noise",
        'cluster_id': cl,
        'cluster_confidence': conf,
        'umap_x': embedding_2d[i, 0],
        'umap_y': embedding_2d[i, 1]
    } for i, (r, cl, conf) in enumerate(zip(records, cluster_labels, confidences))])
    
    df.to_csv(output_dir / "cluster_results.csv", index=False)
    print(f"  ✓ cluster_results.csv ({len(df)} rows)")
    
    # 2. Save Features (for Drill-Down)
    if features is not None:
        np.save(output_dir / "features.npy", features)
        print(f"  ✓ features.npy (Saved for sub-clustering)")

    # 3. Save Summary
    with open(output_dir / "cluster_summary.txt", 'w') as f:
        f.write(f"Clusters found: {len([s for s in summary if s['cluster_id'] >= 0])}\n")
        for info in summary:
            f.write(f"\n{info['cluster'].upper()} (n={info['size']})\n")
            f.write(f"  Dominant: {info['dominant_period']} ({info['purity']*100:.1f}%)\n")
            f.write(f"  Breakdown: {info['period_breakdown']}\n")
    print(f"  ✓ cluster_summary.txt")
    
    # 4. Save JSON
    with open(output_dir / "cluster_data.json", 'w') as f:
        json.dump({'clusters': summary}, f, indent=2, default=str)
    
    return df

def generate_html_visualization(df, records, cluster_labels, summary, output_dir):
    """Generate HTML grid with relative paths."""
    output_dir = Path(output_dir)
    print("  Generating HTML visualization...")

    # Read assignment_method (and possibly updated cluster_id) from CSV if present
    assignment_map = {}
    clusterid_map = {}
    if df is None:
        csv_path = output_dir / "cluster_results.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                df = None
    if df is not None and 'assignment_method' in df.columns:
        try:
            for _, row in df[['id', 'assignment_method', 'cluster_id']].iterrows():
                assignment_map[int(row['id'])] = str(row['assignment_method'])
                clusterid_map[int(row['id'])] = int(row['cluster_id'])
        except Exception:
            assignment_map = {}
            clusterid_map = {}

    clusters_html = []
    for info in summary:
        cluster_id = info['cluster_id']
        cluster_items = []
        for i, (r, cl) in enumerate(zip(records, cluster_labels)):
            coin_id = int(r['id'])
            effective_cluster_id = clusterid_map.get(coin_id, cl)
            if effective_cluster_id == cluster_id:
                cluster_items.append((r, i))

        # Sample 20 images
        import random
        sample = random.sample(cluster_items, min(20, len(cluster_items)))

        images_html = []
        for record, _ in sample:
            path = resolve_image_path(record['image_path'])
            if path:
                # Force absolute resolution before computing relpath
                # This prevents broken links like cluster_output/trivalaya_data/...
                try:
                    abs_path = Path(path).resolve()
                    abs_output = Path(output_dir).resolve()
                    rel_path = os.path.relpath(abs_path, abs_output)
                except ValueError:
                    rel_path = str(Path(path).resolve())  # Fallback if on different drive

                coin_id = int(record['id'])
                method = assignment_map.get(coin_id)
                if method is None:
                    method = "clustered" if cluster_id >= 0 else "noise"

                images_html.append(f'''
                    <div class="coin-card {method}">
                        <img src="{rel_path}" alt="coin" loading="lazy">
                        <div class="coin-info"><span class="period">{record['period']}</span></div>
                    </div>
                ''')

        clusters_html.append(f'''
        <div class="cluster" id="cluster-{cluster_id}">
            <div class="cluster-header">
                <h2>{info['cluster']} ({info['size']})</h2>
                <div class="purity">Dominant: {info['dominant_period']}</div>
            </div>
            <div class="coin-grid">{''.join(images_html)}</div>
        </div>
        ''')

    html = f'''<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ background: #1a1a2e; color: #eee; font-family: sans-serif; padding: 20px; }}
        .cluster {{ background: #16213e; margin-bottom: 30px; border-radius: 8px; overflow: hidden; }}
        .cluster-header {{ padding: 15px; border-bottom: 1px solid #333; }}
        .coin-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 10px; padding: 15px; }}
        .coin-card img {{ width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 4px; }}
        .coin-card.knn_assigned {{
            border: 2px dashed #ffa500;  /* orange dashed border */
        }}
        .coin-card.clustered {{
            border: 2px solid transparent;
        }}
        .coin-card.noise {{
            border: 2px solid #ff4444;  /* red border for remaining noise */
        }}
        h2 {{ color: #ffd700; margin: 0; }}
    </style>
</head>
<body>
    <h1>🪙 Trivalaya Visual Clusters</h1>
    <div class="legend">
        <span class="legend-item"><span class="swatch clustered"></span> Clustered</span>
        <span class="legend-item"><span class="swatch knn"></span> KNN Assigned</span>
    </div>
    {''.join(clusters_html)}
</body>
</html>'''

    with open(output_dir / "cluster_visualization.html", 'w') as f:
        f.write(html)
    print(f"  ✓ cluster_visualization.html")

def drill_down_cluster(target_cluster_id, features, df, output_dir):
    """Isolate one cluster and break it into sub-clusters."""
    print(f"\n⛏️ DRILLING DOWN into Cluster {target_cluster_id}...")
    
    cluster_mask = df['cluster_id'] == target_cluster_id
    indices = df.index[cluster_mask].tolist()
    
    if not indices:
        print(f"❌ Cluster {target_cluster_id} not found or empty.")
        return

    sub_features = features[indices]
    sub_records = df.loc[cluster_mask].to_dict('records')
    
    # Fix key naming mismatch
    for r in sub_records:
        if 'parser_period' in r: r['period'] = r.pop('parser_period')
    
    print(f"   Analyzing {len(sub_features)} coins in sub-space...")

    import umap
    print("   Re-calculating UMAP (local)...")
    reducer = umap.UMAP(n_components=5, n_neighbors=100, min_dist=0.2, metric='cosine', random_state=42)
    sub_embedding = reducer.fit_transform(sub_features)

    import hdbscan
    print("   Re-clustering (sensitive)...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2, cluster_selection_method='leaf')
    sub_labels = clusterer.fit_predict(sub_embedding)
    
    # Save Sub-Report
    sub_output_dir = Path(output_dir) / f"subcluster_{target_cluster_id}"
    sub_output_dir.mkdir(exist_ok=True)
    
    viz_reducer = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.1)
    sub_embedding_2d = viz_reducer.fit_transform(sub_features)
    sub_confidences = np.ones(len(sub_labels))
    
    _, sub_summary = analyze_clusters(sub_records, sub_labels, sub_confidences)
    
    save_results(sub_records, sub_labels, sub_confidences, sub_embedding_2d, sub_summary, sub_output_dir, features=sub_features)
    generate_html_visualization(None, sub_records, sub_labels, sub_summary, sub_output_dir)
    print(f"\n✅ Sub-clustering complete! View results in: {sub_output_dir}/cluster_visualization.html")

def main():
    parser = argparse.ArgumentParser(description="Visual clustering of coin images")
    parser.add_argument('--sample', type=int, help="Sample N images")
    parser.add_argument('--min-cluster', type=int, default=20, help="Min cluster size")
    parser.add_argument('--min-samples', type=int, default=10, help="HDBSCAN min_samples")
    parser.add_argument('--gpu', action='store_true', help="Use GPU")
    parser.add_argument('--output', type=str, default='cluster_output', help="Output dir")
    parser.add_argument('--input_dir', type=str, help='Cluster images in this folder')
    parser.add_argument('--drill-down', type=int, help="ID of cluster to split")
    
    # NEW: Precomputed embeddings support (v5.2)
    parser.add_argument('--embeddings', type=str, help="Path to .npy embeddings to use instead of CLIP")
    parser.add_argument('--embeddings-meta', type=str, help="Path to embeddings meta.json (paths/labels)")
    parser.add_argument('--skip-umap', action='store_true', help="Cluster on raw embeddings (1280-d) instead of UMAP projection")
    parser.add_argument('--pca', type=int, metavar='DIMS', help="Reduce to N dims via PCA before clustering (e.g., --pca 50)")
    
    args = parser.parse_args()
    
    print("🪙 TRIVALAYA VISUAL CLUSTERING")
    print("=" * 50)
    
    # --- DRILL DOWN MODE ---
    if args.drill_down is not None:
        output_dir = Path(args.output)
        feature_path = output_dir / "features.npy"
        results_path = output_dir / "cluster_results.csv"
        
        if not feature_path.exists() or not results_path.exists():
            print("❌ Missing features.npy or cluster_results.csv. Run a full cluster first.")
            return
            
        print(f"📂 Loading previous results from {output_dir}...")
        features = np.load(feature_path)
        df = pd.read_csv(results_path)
        
        drill_down_cluster(args.drill_down, features, df, args.output)
        return

    # --- STANDARD MODE ---
    if args.input_dir and os.path.exists(args.input_dir):
        print(f"🚀 MODE: Directory Clustering ({args.input_dir})")
        records = load_dataset_from_directory(args.input_dir, sample_size=args.sample)
    else:
        print("🚀 MODE: Database Clustering")
        records = load_dataset_from_db(sample_size=args.sample)
    
    if not records:
        print("❌ No records found.")
        return
    
    # --- FEATURE EXTRACTION: Precomputed vs CLIP ---
    if args.embeddings:
        if not args.embeddings_meta:
            raise ValueError("--embeddings-meta is required when using --embeddings")
        features, valid_records = load_precomputed_features(
            records,
            args.embeddings,
            args.embeddings_meta
        )
        # If using precomputed embeddings, default to clustering in original space
        # (UMAP is great for visualization, but often increases "noise" for HDBSCAN)
        # PCA takes precedence if specified
        if args.pca:
            print(f"  ℹ️  Using PCA reduction to {args.pca} dims for clustering.")
        ##elif not args.skip_umap:
        ##    print("  ℹ️  Defaulting to --skip-umap because --embeddings is in use.")
        ##    args.skip_umap = True
    else:
        model, preprocess, device, model_type = load_clip_model(use_gpu=args.gpu)
        features, valid_records = extract_features(records, model, preprocess, device, model_type)
    
    if len(features) < 5:
        print("❌ Too few valid images")
        return
    
    cluster_labels, confidences, embedding_2d = cluster_features(
        features, args.min_cluster, args.min_samples, 
        skip_umap=args.skip_umap, pca_dims=args.pca
    )
    _, summary = analyze_clusters(valid_records, cluster_labels, confidences)
    
    suggestions = suggest_cluster_names(summary)
    print("\n💡 Suggested cluster names:")
    for cluster_id, name in suggestions.items():
        print(f"   Cluster {cluster_id}: {name}")
    
    save_results(valid_records, cluster_labels, confidences, embedding_2d, summary, args.output, features=features)
    generate_html_visualization(None, valid_records, cluster_labels, summary, args.output)
    
    print(f"\n✅ Done! Review results in {args.output}/")

if __name__ == "__main__":
    main()