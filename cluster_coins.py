#!/usr/bin/env python3
"""
cluster_coins.py - Visual clustering of coin images using CLIP embeddings.

Extracts features from coin images, clusters them visually, and compares
against parser-assigned periods to identify natural visual groupings.

Usage:
    python cluster_coins.py                    # Full run
    python cluster_coins.py --sample 1000      # Quick test with 1000 images
    python cluster_coins.py --min-cluster 30   # Require 30+ images per cluster
    python cluster_coins.py --gpu              # Use GPU if available

Outputs:
    - cluster_results.csv: image_path, parser_period, visual_cluster, confidence
    - cluster_visualization.html: Interactive grid view of clusters
    - cluster_summary.txt: Statistics and cluster composition

Requirements:
    pip install torch torchvision pillow numpy pandas umap-learn hdbscan tqdm
    pip install git+https://github.com/openai/CLIP.git
    # OR for newer systems:
    pip install open-clip-torch
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

# --- CONFIGURATION ---
DB_HOST = os.getenv("TRIVALAYA_DB_HOST", "127.0.0.1")
DB_USER = os.getenv("TRIVALAYA_DB_USER", "auction_user")
DB_PASSWORD = os.getenv("TRIVALAYA_DB_PASSWORD", "Veritas@2024")
DB_NAME = os.getenv("TRIVALAYA_DB_NAME", "auction_data")

OUTPUT_DIR = Path("cluster_output")
BATCH_SIZE = 32  # Images per batch for feature extraction


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
    
    # UPDATED QUERY: Added 'AND m.is_active = 1'
    query = """
        SELECT m.id, m.image_path, m.period, m.label_confidence, m.raw_label
        FROM ml_dataset m
        WHERE m.image_path IS NOT NULL 
        AND m.image_path != ''
        AND m.is_active = 1  -- <--- CRITICAL FILTER
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
    """Load images directly from a folder (for re-clustering subsets)."""
    print(f"📂 Scanning directory: {directory}...")
    directory = Path(directory)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif'}
    records = []
    
    # Walk through directory
    files = [f for f in directory.rglob("*") if f.suffix.lower() in image_extensions]
    
    if sample_size:
        import random
        random.shuffle(files)
        files = files[:sample_size]
        
    for i, file_path in enumerate(files):
        # Create a "fake" record structure to match the DB format
        records.append({
            'id': i,
            'image_path': str(file_path.absolute()), # Use absolute path
            'period': 'unknown',       # We don't have metadata for these
            'label_confidence': 0.0,
            'raw_label': file_path.name,
            'coin_likelihood': 1.0,
            'edge_support': 1.0
        })
        
    print(f"  Found {len(records)} images in directory")
    return records

def resolve_image_path(db_path):
    """Find actual file location."""
    path_obj = Path(db_path)
    
    # If it's already an absolute path that exists, return it immediately
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
    """Load CLIP model for feature extraction."""
    print("🧠 Loading CLIP model...")
    
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    
    # Try open_clip first (more reliable install)
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        model = model.to(device)
        model.eval()
        print("  Loaded open_clip ViT-B-32")
        return model, preprocess, device, "open_clip"
    except ImportError:
        pass
    
    # Fall back to original CLIP
    try:
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        print("  Loaded OpenAI CLIP ViT-B/32")
        return model, preprocess, device, "clip"
    except ImportError:
        pass
    
    # Final fallback: torchvision ResNet
    print("  ⚠️  CLIP not available, using ResNet50 features")
    import torchvision.models as models
    import torchvision.transforms as transforms
    
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classifier
    model = model.to(device)
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return model, preprocess, device, "resnet"


def extract_features(records, model, preprocess, device, model_type):
    """Extract visual features from all images."""
    print(f"\n🔍 Extracting features from {len(records)} images...")
    
    features = []
    valid_records = []
    failed = 0
    
    # Process in batches
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
            
            # Process batch
            if len(batch_images) >= BATCH_SIZE:
                batch_features = process_batch(batch_images, model, device, model_type)
                features.extend(batch_features)
                valid_records.extend(batch_records)
                batch_images = []
                batch_records = []
                
        except Exception as e:
            failed += 1
            continue
    
    # Process remaining
    if batch_images:
        batch_features = process_batch(batch_images, model, device, model_type)
        features.extend(batch_features)
        valid_records.extend(batch_records)
    
    print(f"  Extracted: {len(features)}, Failed: {failed}")
    
    return np.array(features), valid_records


def process_batch(images, model, device, model_type):
    """Process a batch of images through the model."""
    import torch
    
    batch = torch.stack(images).to(device)
    
    with torch.no_grad():
        if model_type == "open_clip":
            features = model.encode_image(batch)
        elif model_type == "clip":
            features = model.encode_image(batch)
        else:  # resnet
            features = model(batch).squeeze()
    
    # Normalize features
    features = features.cpu().numpy()
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    return list(features)


def cluster_features(features, min_cluster_size=20, min_samples=10):
    """Cluster features using UMAP + HDBSCAN."""
    print(f"\n🎯 Clustering {len(features)} feature vectors...")
    
    import umap
    import hdbscan
    
    # Dimensionality reduction
    print("  Running UMAP...")
    reducer = umap.UMAP(
        n_components=15,  # Reduce to 15D for clustering
        n_neighbors=30,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
        verbose=False
    )
    embedding = reducer.fit_transform(features)
    
    # Also get 2D for visualization
    print("  Creating 2D projection for visualization...")
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
        verbose=False
    )
    embedding_2d = reducer_2d.fit_transform(features)
    
    # Clustering
    print(f"  Running HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    cluster_labels = clusterer.fit_predict(embedding)
    
    # Get soft cluster assignments (confidence)
    soft_clusters = hdbscan.membership_vector(clusterer, embedding)
    confidences = np.max(soft_clusters, axis=1) if len(soft_clusters.shape) > 1 else np.ones(len(cluster_labels))
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"  Found {n_clusters} clusters, {n_noise} noise points ({n_noise/len(cluster_labels)*100:.1f}%)")
    
    return cluster_labels, confidences, embedding_2d


def analyze_clusters(records, cluster_labels, confidences):
    """Analyze cluster composition and generate statistics."""
    print("\n📈 Analyzing cluster composition...")
    
    cluster_data = defaultdict(list)
    
    for record, cluster, conf in zip(records, cluster_labels, confidences):
        cluster_data[cluster].append({
            'id': record['id'],
            'path': record['image_path'],
            'period': record['period'] or 'unknown',
            'confidence': conf,
            'raw_label': record.get('raw_label', '')[:100]
        })
    
    # Generate summary stats
    summary = []
    for cluster_id in sorted(cluster_data.keys()):
        items = cluster_data[cluster_id]
        periods = Counter(item['period'] for item in items)
        
        dominant_period = periods.most_common(1)[0] if periods else ('unknown', 0)
        purity = dominant_period[1] / len(items) if items else 0
        
        cluster_name = f"cluster_{cluster_id}" if cluster_id >= 0 else "noise"
        
        summary.append({
            'cluster': cluster_name,
            'cluster_id': cluster_id,
            'size': len(items),
            'dominant_period': dominant_period[0],
            'purity': purity,
            'period_breakdown': dict(periods),
            'avg_confidence': np.mean([item['confidence'] for item in items])
        })
    
    # Sort by size
    summary.sort(key=lambda x: x['size'], reverse=True)
    
    return cluster_data, summary


def suggest_cluster_names(summary):
    """Suggest names for clusters based on composition."""
    suggestions = {}
    
    for cluster_info in summary:
        if cluster_info['cluster_id'] < 0:
            continue
            
        periods = cluster_info['period_breakdown']
        dominant = cluster_info['dominant_period']
        purity = cluster_info['purity']
        
        if purity > 0.8:
            # High purity - use period name
            suggestions[cluster_info['cluster_id']] = f"{dominant}_pure"
        elif purity > 0.5:
            # Moderate purity - dominant + mixed
            second = [p for p in periods.keys() if p != dominant]
            second_name = second[0] if second else "mixed"
            suggestions[cluster_info['cluster_id']] = f"{dominant}_with_{second_name}"
        else:
            # Low purity - descriptive
            top_two = sorted(periods.items(), key=lambda x: -x[1])[:2]
            suggestions[cluster_info['cluster_id']] = f"mixed_{'_'.join(p[0] for p in top_two)}"
    
    return suggestions


def save_results(records, cluster_labels, confidences, embedding_2d, summary, output_dir):
    """Save all results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 Saving results to {output_dir}/")
    
    # 1. CSV with all assignments
    df = pd.DataFrame([{
        'id': r['id'],
        'image_path': r['image_path'],
        'parser_period': r['period'] or 'unknown',
        'visual_cluster': f"cluster_{cl}" if cl >= 0 else "noise",
        'cluster_id': cl,
        'cluster_confidence': conf,
        'parser_confidence': r['label_confidence'],
        'umap_x': embedding_2d[i, 0],
        'umap_y': embedding_2d[i, 1],
        'raw_label': r.get('raw_label', '')[:200]
    } for i, (r, cl, conf) in enumerate(zip(records, cluster_labels, confidences))])
    
    df.to_csv(output_dir / "cluster_results.csv", index=False)
    print(f"  ✓ cluster_results.csv ({len(df)} rows)")
    
    # 2. Summary text
    with open(output_dir / "cluster_summary.txt", 'w') as f:
        f.write("TRIVALAYA VISUAL CLUSTERING SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total images: {len(records)}\n")
        n_clusters = len([s for s in summary if s['cluster_id'] >= 0])
        n_noise = sum(1 for cl in cluster_labels if cl < 0)
        f.write(f"Clusters found: {n_clusters}\n")
        f.write(f"Noise points: {n_noise} ({n_noise/len(records)*100:.1f}%)\n\n")
        
        f.write("CLUSTER BREAKDOWN\n")
        f.write("-" * 60 + "\n")
        
        for info in summary:
            f.write(f"\n{info['cluster'].upper()} (n={info['size']})\n")
            f.write(f"  Dominant period: {info['dominant_period']} ({info['purity']*100:.1f}% purity)\n")
            f.write(f"  Avg confidence: {info['avg_confidence']:.2f}\n")
            f.write(f"  Breakdown: {info['period_breakdown']}\n")
    
    print(f"  ✓ cluster_summary.txt")
    
    # 3. JSON for programmatic use
    with open(output_dir / "cluster_data.json", 'w') as f:
        json.dump({
            'generated': datetime.now().isoformat(),
            'total_images': len(records),
            'n_clusters': len([s for s in summary if s['cluster_id'] >= 0]),
            'clusters': summary
        }, f, indent=2, default=str)
    
    print(f"  ✓ cluster_data.json")
    
    return df


def generate_html_visualization(df, records, cluster_labels, summary, output_dir):
    """Generate interactive HTML visualization of clusters."""
    output_dir = Path(output_dir)
    
    print("  Generating HTML visualization...")
    
    # Build cluster groups with sample images
    clusters_html = []
    
    for info in summary:
        cluster_id = info['cluster_id']
        cluster_records = [
            (r, i) for i, (r, cl) in enumerate(zip(records, cluster_labels)) 
            if cl == cluster_id
        ]
        
        # Sample up to 20 images per cluster for display
        import random
        sample = random.sample(cluster_records, min(20, len(cluster_records)))
        
        # Build image grid
        images_html = []
        for record, idx in sample:
            path = resolve_image_path(record['image_path'])
            if path:
                # Use relative path or base64 encode small images
                images_html.append(f'''
                    <div class="coin-card">
                        <img src="{path}" alt="coin" loading="lazy">
                        <div class="coin-info">
                            <span class="period">{record['period'] or 'unknown'}</span>
                        </div>
                    </div>
                ''')
        
        period_bars = ''.join([
            f'<div class="period-bar" style="width: {count/info["size"]*100}%; background: {get_period_color(period)};" title="{period}: {count}"></div>'
            for period, count in sorted(info['period_breakdown'].items(), key=lambda x: -x[1])
        ])
        
        clusters_html.append(f'''
        <div class="cluster" id="cluster-{cluster_id}">
            <div class="cluster-header">
                <h2>{info['cluster']} <span class="count">({info['size']} coins)</span></h2>
                <div class="purity">Dominant: {info['dominant_period']} ({info['purity']*100:.0f}%)</div>
                <div class="period-breakdown">{period_bars}</div>
            </div>
            <div class="coin-grid">
                {''.join(images_html)}
            </div>
        </div>
        ''')
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trivalaya Visual Clusters</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }}
        h1 {{ 
            text-align: center; 
            padding: 20px;
            color: #ffd700;
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 40px;
            padding: 20px;
            background: #16213e;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .stat {{ text-align: center; }}
        .stat-value {{ font-size: 2em; color: #ffd700; }}
        .stat-label {{ color: #888; }}
        
        .cluster {{
            background: #16213e;
            border-radius: 10px;
            margin-bottom: 30px;
            overflow: hidden;
        }}
        .cluster-header {{
            padding: 15px 20px;
            border-bottom: 1px solid #333;
        }}
        .cluster-header h2 {{
            color: #ffd700;
        }}
        .cluster-header .count {{
            color: #888;
            font-weight: normal;
            font-size: 0.8em;
        }}
        .purity {{
            color: #4ade80;
            margin-top: 5px;
        }}
        .period-breakdown {{
            display: flex;
            height: 8px;
            margin-top: 10px;
            border-radius: 4px;
            overflow: hidden;
        }}
        .period-bar {{
            height: 100%;
        }}
        
        .coin-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
            gap: 10px;
            padding: 15px;
        }}
        .coin-card {{
            position: relative;
            aspect-ratio: 1;
            overflow: hidden;
            border-radius: 8px;
            background: #0f0f23;
        }}
        .coin-card img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.2s;
        }}
        .coin-card:hover img {{
            transform: scale(1.1);
        }}
        .coin-info {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0,0,0,0.8);
            padding: 4px;
            font-size: 0.7em;
            text-align: center;
        }}
        .period {{
            color: #ffd700;
        }}
        
        .toc {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: #16213e;
            padding: 15px;
            border-radius: 10px;
            max-height: 80vh;
            overflow-y: auto;
            z-index: 100;
        }}
        .toc h3 {{ margin-bottom: 10px; color: #ffd700; }}
        .toc a {{
            display: block;
            color: #888;
            text-decoration: none;
            padding: 3px 0;
            font-size: 0.9em;
        }}
        .toc a:hover {{ color: #fff; }}
    </style>
</head>
<body>
    <h1>🪙 Trivalaya Visual Clusters</h1>
    
    <div class="stats">
        <div class="stat">
            <div class="stat-value">{len(records)}</div>
            <div class="stat-label">Total Coins</div>
        </div>
        <div class="stat">
            <div class="stat-value">{len([s for s in summary if s['cluster_id'] >= 0])}</div>
            <div class="stat-label">Clusters</div>
        </div>
        <div class="stat">
            <div class="stat-value">{sum(1 for cl in cluster_labels if cl < 0)}</div>
            <div class="stat-label">Noise</div>
        </div>
    </div>
    
    <div class="toc">
        <h3>Clusters</h3>
        {''.join(f'<a href="#cluster-{s["cluster_id"]}">{s["cluster"]} ({s["size"]})</a>' for s in summary)}
    </div>
    
    {''.join(clusters_html)}
    
    <script>
        // Lazy load images
        document.addEventListener('DOMContentLoaded', function() {{
            const observer = new IntersectionObserver((entries) => {{
                entries.forEach(entry => {{
                    if (entry.isIntersecting) {{
                        const img = entry.target;
                        img.src = img.dataset.src;
                        observer.unobserve(img);
                    }}
                }});
            }});
            
            document.querySelectorAll('img[data-src]').forEach(img => observer.observe(img));
        }});
    </script>
</body>
</html>'''
    
    with open(output_dir / "cluster_visualization.html", 'w') as f:
        f.write(html)
    
    print(f"  ✓ cluster_visualization.html")


def get_period_color(period):
    """Get consistent color for each period."""
    colors = {
        'greek': '#22c55e',
        'roman_imperial': '#ef4444',
        'roman_republican': '#f97316',
        'roman_provincial': '#fb923c',
        'byzantine': '#a855f7',
        'islamic': '#06b6d4',
        'persian': '#eab308',
        'medieval': '#6366f1',
        'celtic': '#14b8a6',
        'unknown': '#6b7280',
    }
    return colors.get(period, '#6b7280')


def main():
    parser = argparse.ArgumentParser(description="Visual clustering of coin images")
    parser.add_argument('--sample', type=int, help="Sample N images for quick test")
    parser.add_argument('--min-cluster', type=int, default=20, help="Minimum cluster size")
    parser.add_argument('--min-samples', type=int, default=10, help="HDBSCAN min_samples")
    parser.add_argument('--gpu', action='store_true', help="Use GPU if available")
    parser.add_argument('--output', type=str, default='cluster_output', help="Output directory")
    
    # Modified Input Logic
    parser.add_argument('--input_dir', type=str, default=None, 
                        help='If set, clusters images in this folder instead of DB')
    
    args = parser.parse_args()
    
    global torch
    import torch
    
    print("🪙 TRIVALAYA VISUAL CLUSTERING")
    print("=" * 50)
    
    # --- LOGIC SWITCH: DB vs DIRECTORY ---
    if args.input_dir and os.path.exists(args.input_dir):
        print(f"🚀 MODE: Directory Clustering ({args.input_dir})")
        records = load_dataset_from_directory(args.input_dir, sample_size=args.sample)
    else:
        print("🚀 MODE: Database Clustering")
        records = load_dataset_from_db(sample_size=args.sample)
    
    if not records:
        print("❌ No records found.")
        return
    
    # Load model
    model, preprocess, device, model_type = load_clip_model(use_gpu=args.gpu)
    
    # Extract features
    features, valid_records = extract_features(records, model, preprocess, device, model_type)
    
    if len(features) < 5: # Lowered limit for small tests
        print("❌ Too few valid images for clustering")
        return
    
    # Cluster
    cluster_labels, confidences, embedding_2d = cluster_features(
        features, 
        min_cluster_size=args.min_cluster,
        min_samples=args.min_samples
    )
    
    # Analyze
    cluster_data, summary = analyze_clusters(valid_records, cluster_labels, confidences)
    
    # Suggest names
    suggestions = suggest_cluster_names(summary)
    print("\n💡 Suggested cluster names:")
    for cluster_id, name in suggestions.items():
        print(f"   Cluster {cluster_id}: {name}")
    
    # Save results
    df = save_results(valid_records, cluster_labels, confidences, embedding_2d, summary, args.output)
    
    # Generate HTML
    generate_html_visualization(df, valid_records, cluster_labels, summary, args.output)
    
    print(f"\n✅ Done! Review results in {args.output}/")
    print(f"   Open cluster_visualization.html in a browser to explore clusters visually")

if __name__ == "__main__":
    main()