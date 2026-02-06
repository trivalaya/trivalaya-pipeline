import pandas as pd
import os
import argparse
import boto3
from collections import Counter
from dotenv import dotenv_values

# Load Spaces config
_spaces_cfg = dotenv_values("/etc/trivalaya/spaces.env")
_s3_client = boto3.client(
    's3',
    region_name=_spaces_cfg.get('SPACES_REGION', 'sfo3'),
    endpoint_url=_spaces_cfg.get('SPACES_ENDPOINT', 'https://sfo3.digitaloceanspaces.com'),
    aws_access_key_id=os.getenv('SPACES_KEY') or _spaces_cfg.get('SPACES_KEY'),
    aws_secret_access_key=os.getenv('SPACES_SECRET') or _spaces_cfg.get('SPACES_SECRET'),
)
_spaces_bucket = _spaces_cfg.get('SPACES_BUCKET', 'trivalaya-data')

def presigned_url(key, expires=86400):
    """Generate a presigned URL for a Spaces object (default 24h)."""
    if not key:
        return ''
    return _s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': _spaces_bucket, 'Key': key},
        ExpiresIn=expires,
    )

def get_period_color(period):
    colors = {
        'greek': '#22c55e', 'roman_imperial': '#ef4444', 'roman_republican': '#f97316',
        'roman_provincial': '#fb923c', 'byzantine': '#a855f7', 'islamic': '#06b6d4',
        'persian': '#eab308', 'medieval': '#6366f1', 'celtic': '#14b8a6', 'unknown': '#6b7280',
    }
    return colors.get(str(period).lower(), '#6b7280')

def resolve_image_url(path_str):
    """Generate a presigned Spaces URL for the image."""
    if not path_str or pd.isna(path_str):
        return ''
    # Strip any leading ./ or / to get the clean Spaces key
    key = str(path_str).lstrip('./')
    return presigned_url(key)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='cluster_output/cluster_results.csv')
    parser.add_argument('--output', default='cluster_output')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"❌ Error: Could not find CSV at {args.csv}")
        return

    print(f"📊 Reading {args.csv}...")
    df = pd.read_csv(args.csv)
    df['cluster_id'] = df['cluster_id'].astype(str)

    # 1. Aggregate Stats
    clusters = []
    unique_ids = df['cluster_id'].unique()
    
    for c_id in unique_ids:
        c_df = df[df['cluster_id'] == c_id]
        
        # Check for noise
        is_noise = (str(c_id) == '-1' or 'noise' in str(c_id))
        cluster_name = f"Cluster {c_id}" if not is_noise else "Noise"
        
        # Stats
        periods = c_df['parser_period'].fillna('unknown').tolist()
        period_counts = Counter(periods)
        dominant = period_counts.most_common(1)[0] if period_counts else ('unknown', 0)
        purity = dominant[1] / len(periods) if periods else 0
        
        clusters.append({
            'id': c_id,
            'name': cluster_name,
            'size': len(c_df),
            'dominant': dominant[0],
            'purity': purity,
            'breakdown': period_counts,
            'images': c_df.sample(min(20, len(c_df)))
        })

    # Sort: Largest first
    clusters.sort(key=lambda x: x['size'], reverse=True)

    # 2. Print Console Summary
    print("\n" + "="*60)
    print(f"{'ID':<12} | {'Count':<8} | {'Dominant Period':<20} | {'Purity':<6}")
    print("-" * 60)
    for c in clusters:
        print(f"{c['id']:<12} | {c['size']:<8} | {c['dominant']:<20} | {c['purity']*100:.0f}%")
    print("="*60 + "\n")

    # 3. Build HTML
    print("   Building HTML Dashboard...")
    
    html_parts = ["""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Trivalaya Clusters</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #cbd5e1; padding: 20px; margin: 0; }
        a { color: inherit; text-decoration: none; }
        
        /* Summary Dashboard */
        .dashboard { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin-bottom: 40px; }
        .stat-card { background: #1e293b; padding: 10px; border-radius: 6px; border-left: 4px solid #3b82f6; transition: transform 0.1s; }
        .stat-card:hover { transform: translateY(-2px); background: #334155; }
        .stat-card h3 { margin: 0; font-size: 1.1em; color: #fff; }
        .stat-card .count { font-size: 1.5em; font-weight: bold; color: #60a5fa; }
        .stat-card .meta { font-size: 0.8em; color: #94a3b8; }
        
        /* Cluster Sections */
        .cluster { background: #1e293b; margin-bottom: 30px; padding: 20px; border-radius: 12px; }
        .cluster-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 10px; margin-bottom: 15px; }
        .cluster h2 { margin: 0; color: #facc15; }
        
        /* Images */
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 8px; }
        .card { aspect-ratio: 1; overflow: hidden; border-radius: 4px; background: #000; position: relative; }
        .card img { width: 100%; height: 100%; object-fit: cover; transition: opacity 0.2s; }
        .card:hover img { opacity: 0.8; }
        .label { position: absolute; bottom: 0; width: 100%; background: rgba(0,0,0,0.7); font-size: 10px; text-align: center; padding: 2px 0; }
        
        /* Bars */
        .bar-container { height: 8px; display: flex; border-radius: 4px; overflow: hidden; margin-bottom: 15px; background: #334155; }
    </style>
</head>
<body>
    <h1>Visual Cluster Dashboard</h1>
    <p style="color:#94a3b8">Total Clusters: """ + str(len(clusters)) + """</p>
    
    <div class="dashboard">
"""]

    # Add Summary Cards
    for c in clusters:
        html_parts.append(f"""
        <a href="#cluster-{c['id']}" class="stat-card">
            <h3>{c['name']}</h3>
            <div class="count">{c['size']}</div>
            <div class="meta">{c['dominant']} ({c['purity']*100:.0f}%)</div>
        </a>
        """)

    html_parts.append("""</div>
    
    <div id="main-content">
    """)

    # Add Full Cluster Views
    for c in clusters:
        # Period Bar
        bar_html = ""
        for p, count in c['breakdown'].most_common():
            width = (count / c['size']) * 100
            color = get_period_color(p)
            bar_html += f'<div style="width:{width}%; background:{color}" title="{p}: {count}"></div>'

        # Images
        img_html = ""
        for _, row in c['images'].iterrows():
            obv_path = row.get('obv_path', '')
            rev_path = row.get('rev_path', '')
            period = row.get('parser_period', 'unknown')
            
            if pd.notna(obv_path) and obv_path and pd.notna(rev_path) and rev_path:
                obv_url = resolve_image_url(obv_path)
                rev_url = resolve_image_url(rev_path)
                img_html += f"""
                <div class="card" title="{period}" style="aspect-ratio: 2/1; grid-column: span 2;">
                    <img src="{obv_url}" loading="lazy" style="width:50%; float:left; height:100%; object-fit:cover;">
                    <img src="{rev_url}" loading="lazy" style="width:50%; float:left; height:100%; object-fit:cover;">
                    <div class="label">{period}</div>
                </div>"""
            else:
                # Fallback: single image
                img_path = obv_path or rev_path or row.get('image_path', '')
                img_url = resolve_image_url(img_path)
                if img_url:
                    img_html += f"""
                <div class="card" title="{period}">
                    <img src="{img_url}" loading="lazy">
                    <div class="label">{period}</div>
                </div>"""

        html_parts.append(f"""
        <div class="cluster" id="cluster-{c['id']}">
            <div class="cluster-header">
                <h2>{c['name']}</h2>
                <span style="color:#94a3b8">{c['size']} coins</span>
            </div>
            <div class="bar-container">{bar_html}</div>
            <div class="grid">{img_html}</div>
        </div>
        """)

    html_parts.append("</div></body></html>")

    out_path = os.path.join(args.output, 'cluster_visualization.html')
    with open(out_path, 'w') as f:
        f.write("\n".join(html_parts))
    
    print(f"✅ Dashboard generated: {out_path}")

if __name__ == "__main__":
    main()