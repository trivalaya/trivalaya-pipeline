#!/usr/bin/env python3
"""
export_coin_embeddings.py
"The Bridge Version"
- Links 'processed/vision/...' manifest paths to 'leu_1_...' embedding paths.
- Uses a canonical key (auction_sale_lot) to bridge the two datasets.
- Auto-corrects 'coin_id' vs 'coin_identity'.
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

# --- THE BRIDGE LOGIC ---
def get_canonical_key(path_str):
    """
    Extracts a common ID (e.g., 'leu_1_01006') from distinct path formats.
    """
    if not path_str:
        return ""
    
    s = str(path_str).replace("\\", "/")
    
    # Strategy 1: Look for "Folder Structure" (Manifest style)
    # CHANGED: (\d+) -> ([a-zA-Z0-9]+) to handle sale IDs like "t23" or "XXIX"
    match_manifest = re.search(r'/([a-zA-Z]+)/([a-zA-Z0-9]+)/Lot_(\d+)', s)
    if match_manifest:
        auct, sale, lot = match_manifest.groups()
        return f"{auct.lower()}_{sale}_{lot}"

    # Strategy 2: Look for "Filename Structure" (Embeddings style)
    match_filename = re.search(r'([a-zA-Z]+)_([a-zA-Z0-9]+)_(\d{4,5})', Path(s).name)
    if match_filename:
        auct, sale, lot = match_filename.groups()
        return f"{auct.lower()}_{sale}_{lot}"

    return Path(s).stem

# ------------------------

def load_side_embeddings(embeddings_path: str, meta_path: str):
    """Load per-side embeddings and metadata."""
    print(f"📦 Loading side embeddings from {embeddings_path}...")
    
    vectors = np.load(embeddings_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    paths = meta['paths']
    confidences = meta.get('confidences', [0.0] * len(paths))
    
    if len(paths) != vectors.shape[0]:
        raise ValueError(f"Mismatch: {vectors.shape[0]} vectors vs {len(paths)} paths")
    
    print(f"  Loaded {vectors.shape[0]} side embeddings")
    return vectors, paths, confidences

def load_manifest_with_coin_identity(manifest_path: str):
    print(f"📄 Loading manifest from {manifest_path}...")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    if not manifest:
        return []

    # Auto-fix coin_id -> coin_identity
    fixed = 0
    for r in manifest:
        if 'coin_identity' not in r and 'coin_id' in r:
            r['coin_identity'] = r['coin_id']
            fixed += 1
            
    if fixed:
        print(f"  🔧 Auto-corrected {fixed} records (coin_id -> coin_identity)")

    valid = [r for r in manifest if r.get('coin_identity')]
    return valid

def build_path_index(paths: list, confidences: list) -> dict:
    """Build index using the Canonical Key (The Bridge)."""
    index = {}
    
    print("  🏗️  Building canonical index...")
    for i, p in enumerate(paths):
        key = get_canonical_key(p)
        if not key: continue

        # If duplicate keys exist (e.g. leu_1_01006 appeared twice),
        # prefer the one with higher confidence or later index?
        # Let's keep the first one found for simplicity, or overwrite?
        # Typically overwriting with the "cleanest" is better, but let's just store mapping.
        index[key] = i
        
    print(f"  ✅ Indexed {len(index)} unique canonical keys")
    return index

def group_by_coin_identity(manifest: list, vectors: np.ndarray, path_index: dict, confidences: list):
    print("🔗 Grouping embeddings by coin_identity...")

    raw_groups = defaultdict(lambda: {'sides': defaultdict(list)})
    matched_count = 0
    unmatched_count = 0
    unmatched_examples = []

    for manifest_idx, record in enumerate(manifest):
        coin_id = record.get('coin_identity')
        if not coin_id: continue

        # Collect paths to check
        candidates = []
        # Check Pair Format
        if record.get('obv_path'): candidates.append(('obverse', record['obv_path']))
        if record.get('rev_path'): candidates.append(('reverse', record['rev_path']))
        # Check Single Format
        if record.get('image_path'): candidates.append((record.get('side', 'unknown'), record['image_path']))

        for side, raw_path in candidates:
            # APPLY THE BRIDGE
            key = get_canonical_key(raw_path)
            
            if key not in path_index:
                unmatched_count += 1
                if len(unmatched_examples) < 3:
                    unmatched_examples.append(f"{raw_path} -> Key: {key}")
                continue

            embedding_idx = path_index[key]
            matched_count += 1

            quality = confidences[embedding_idx] if embedding_idx < len(confidences) else 0.0

            raw_groups[coin_id]['sides'][side].append({
                'embedding': vectors[embedding_idx],
                'path': raw_path,
                'period': record.get('period'),
                'quality': quality
            })

    print(f"  📊 Match Stats: {matched_count} matched sides, {unmatched_count} unmatched paths")
    if unmatched_count > 0:
        print(f"  ⚠️  Unmatched examples: {unmatched_examples}")

    # Resolve duplicates
    cleaned_coins = {}
    for coin_id, data in raw_groups.items():
        clean_sides = {}
        periods = set()
        
        for side, items in data['sides'].items():
            # Pick best quality
            best = max(items, key=lambda x: x['quality'])
            clean_sides[side] = best
            if best['period']: periods.add(best['period'])
            
        final_period = sorted(list(periods))[0] if periods else None
        cleaned_coins[coin_id] = {'sides': clean_sides, 'period': final_period}

    return cleaned_coins

def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / (norm + 1e-12)

def export_coin_embeddings(coins: dict, output_dir: Path):
    print("\n📤 Exporting coin-level embeddings...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_coins = []
    
    for coin_id, data in coins.items():
        sides = data['sides']
        
        # Gather vectors
        vecs = []
        if 'obverse' in sides: vecs.append(sides['obverse']['embedding'])
        if 'reverse' in sides: vecs.append(sides['reverse']['embedding'])
        if not vecs and 'unknown' in sides: vecs.append(sides['unknown']['embedding'])
        
        if not vecs: continue
        
        # Average
        vecs = [normalize_vector(v) for v in vecs]
        avg_emb = normalize_vector(np.mean(vecs, axis=0))
        
        all_coins.append({
            'coin_identity': coin_id,
            'embedding_avg': avg_emb,
            'period': data['period'],
            'sides_available': list(sides.keys())
        })

    if all_coins:
        np.save(output_dir / "coin_embeddings_avg.npy", np.array([c['embedding_avg'] for c in all_coins], dtype=np.float32))
        with open(output_dir / "coin_embeddings_avg_meta.json", 'w') as f:
            json.dump({
                'coin_identities': [c['coin_identity'] for c in all_coins],
                'periods': [c['period'] for c in all_coins],
                'sides_available': [c['sides_available'] for c in all_coins]
            }, f, indent=2)

    print(f"  ✅ Processed {len(all_coins)} coins.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--embeddings-meta', required=True)
    parser.add_argument('--output', default='coin_embeddings')
    args = parser.parse_args()
    
    vectors, paths, confs = load_side_embeddings(args.embeddings, args.embeddings_meta)
    manifest = load_manifest_with_coin_identity(args.manifest)
    path_index = build_path_index(paths, confs)
    coins = group_by_coin_identity(manifest, vectors, path_index, confs)
    export_coin_embeddings(coins, args.output)

if __name__ == "__main__":
    main()