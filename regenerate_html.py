import pandas as pd
import os
import argparse
from pathlib import Path
from collections import Counter

def get_period_color(period):
    colors = {
        'greek': '#22c55e', 'roman_imperial': '#ef4444', 'roman_republican': '#f97316',
        'roman_provincial': '#fb923c', 'byzantine': '#a855f7', 'islamic': '#06b6d4',
        'persian': '#eab308', 'medieval': '#6366f1', 'celtic': '#14b8a6', 'unknown': '#6b7280',
    }
    return colors.get(str(period).lower(), '#6b7280')

def resolve_image_path(path_str):
    # Quick fix to make sure local server can find the images
    # We strip absolute paths to make them relative if they are inside the current dir
    cwd = os.getcwd()
    if path_str.startswith(cwd):
        return os.path.relpath(path_str, cwd)
    return path_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='cluster_output/cluster_results.csv')
    parser.add_argument('--output', default='cluster_output')
    args = parser.parse_args()

    print(f"📊 Reading {args.csv}...")
    df = pd.read_csv(args.csv)
    
    # Ensure cluster_id is string to handle "5-0"
    df['cluster_id'] = df['cluster_id'].astype(str)

    # 1. Aggregate Stats by Cluster
    print("   Calculating cluster stats...")
    clusters = []
    
    # Get unique cluster IDs (excluding noise if you want, but usually good to show)
    unique_ids = df['cluster_id'].unique()
    
    for c_id in unique_ids:
        # Filter rows for this cluster
        c_df = df[df['cluster_id'] == c_id]
        
        # Skip noise if desired, or handle differently
        is_noise = (str(c_id) == '-1' or 'noise' in str(c_id))
        
        # Period stats
        periods = c_df['parser_period'].fillna('unknown').tolist()
        period_counts = Counter(periods)
        dominant = period_counts.most_common(1)[0] if period_counts else ('unknown', 0)
        purity = dominant[1] / len(periods) if periods else 0
        
        clusters.append({
            'id': c_id,
            'name': f"Cluster {c_id}" if not is_noise else "Noise",
            'size': len(c_df),
            'dominant': dominant[0],
            'purity': purity,
            'breakdown': period_counts,
            'images': c_df.sample(min(20, len(c_df))) # Pick 20 random sample images
        })

    # Sort: Largest clusters first
    clusters.sort(key=lambda x: x['size'], reverse=True)

    # 2. Build HTML
    print("   Building HTML...")
    
    # --- HTML Header ---
    html_parts = ["""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Trivalaya Updated Clusters</title>
    <style>
        body { font-family: sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
        .cluster { background: #16213e; margin-bottom: 30px; padding: 15px; border-radius: 8px; }
        .cluster h2 { color: #ffd700; border-bottom: 1px solid #333; padding-bottom: 10px; }
        .purity { color: #4ade80; font-size: 0.9em; margin-bottom: 10px; }
        .bar { height: 6px; display: flex; border-radius: 3px; overflow: hidden; margin-bottom: 15px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 10px; }
        .card { position: relative; aspect-ratio: 1; overflow: hidden; border-radius: 4px; background: #000; }
        .card img { width: 100%; height: 100%; object-fit: cover; transition: transform 0.2s; }
        .card:hover img { transform: scale(1.1); }
        .label { position: absolute; bottom: 0; background: rgba(0,0,0,0.7); width: 100%; font-size: 0.7em; text-align: center; padding: 2px; }
    </style>
</head>
<body>
    <h1>Updated Visual Clusters</h1>
"""]

    # --- Cluster Blocks ---
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
            img_path = resolve_image_path(row['image_path'])
            img_html += f"""
                <div class="card">
                    <img src="{img_path}" loading="lazy">
                    <div class="label">{row['parser_period']}</div>
                </div>"""

        html_parts.append(f"""
        <div class="cluster" id="cluster-{c['id']}">
            <h2>{c['name']} <span style="font-size:0.6em; color:#888">({c['size']} coins)</span></h2>
            <div class="purity">Dominant: {c['dominant']} ({c['purity']*100:.0f}%)</div>
            <div class="bar">{bar_html}</div>
            <div class="grid">{img_html}</div>
        </div>
        """)

    html_parts.append("</body></html>")

    # 3. Write File
    out_path = os.path.join(args.output, 'cluster_visualization.html')
    with open(out_path, 'w') as f:
        f.write("\n".join(html_parts))
    
    print(f"✅ Visualization updated: {out_path}")

if __name__ == "__main__":
    main()