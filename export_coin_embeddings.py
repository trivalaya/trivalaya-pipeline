#!/usr/bin/env python3
"""
export_coin_embeddings.py - Generate coin-level embeddings from side embeddings.
ROCK-SOLID VERSION:
- Strict determinism (sorted sets)
- Defensive normalization (pre-average)
- Unmatched path sampling for debugging
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_side_embeddings(embeddings_path: str, meta_path: str):
    """Load per-side embeddings and metadata."""
    print(f"📦 Loading side embeddings from {embeddings_path}...")
    
    vectors = np.load(embeddings_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    paths = meta['paths']
    # If side_confidence isn't in meta, we'll default to 0 later
    confidences = meta.get('confidences', [0.0] * len(paths))
    
    if len(paths) != vectors.shape[0]:
        raise ValueError(f"Mismatch: {vectors.shape[0]} vectors vs {len(paths)} paths")
    
    print(f"  Loaded {vectors.shape[0]} side embeddings")
    return vectors, paths, confidences


def load_manifest_with_coin_identity(manifest_path: str):
    """Load manifest and validate required fields."""
    print(f"📄 Loading manifest from {manifest_path}...")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    valid = [r for r in manifest if r.get('coin_identity')]
    if len(valid) < len(manifest):
        print(f"  ⚠️  Skipped {len(manifest) - len(valid)} records missing coin_identity")
        
    return valid


def build_path_index(paths: list) -> dict:
    """
    Build path -> embedding index lookup.
    STRICT MODE: Raises error on collision.
    """
    index = {}
    for i, p in enumerate(paths):
        # Normalize: unified separators, no leading dot-slash
        normalized = str(p).replace("\\", "/").strip().lstrip("./")
        
        # Check for collision
        if normalized in index and index[normalized] != i:
            raise ValueError(f"❌ FATAL: Duplicate path collision in embeddings: '{normalized}'. "
                             "Check your embedding generation process for duplicates.")
        
        index[normalized] = i
    return index


def group_by_coin_identity(manifest: list, vectors: np.ndarray, path_index: dict, confidences: list):
    """
    Group by coin_identity, resolving duplicates by quality score.
    """
    print("🔗 Grouping embeddings by coin_identity...")
    
    # Temporary storage: coin_id -> side -> list of candidates
    raw_groups = defaultdict(lambda: {'sides': defaultdict(list)})
    
    matched_count = 0
    unmatched_count = 0
    unmatched_examples = []
    
    # Use enumerate to get the TRUE manifest index for stable tie-breaking
    for manifest_idx, record in enumerate(manifest):
        coin_id = record.get('coin_identity')
        side = record.get('side', 'unknown')
        path = record.get('image_path', '')
        period = record.get('period')
        
        # Normalize path to match index
        normalized_path = str(path).replace("\\", "/").strip().lstrip("./")
        
        if normalized_path not in path_index:
            unmatched_count += 1
            if len(unmatched_examples) < 5:
                unmatched_examples.append(normalized_path)
            continue
            
        embedding_idx = path_index[normalized_path]
        matched_count += 1
        
        # Get quality score (prefer explicit confidence, fallback to 0)
        quality = confidences[embedding_idx] if embedding_idx < len(confidences) else 0.0
        
        if 'side_confidence' in record and record['side_confidence'] is not None:
            quality = float(record['side_confidence'])

        raw_groups[coin_id]['sides'][side].append({
            'embedding': vectors[embedding_idx],
            'path': path,
            'period': period,
            'quality': quality,
            'manifest_idx': manifest_idx  # Stable tie-break using input order
        })

    print(f"  📊 Match Stats: {matched_count} matched, {unmatched_count} unmatched")
    if unmatched_count > 0:
        print(f"  ⚠️  {unmatched_count} manifest records had no corresponding embedding.")
        print(f"  Examples: {unmatched_examples}")

    # Resolve duplicates and conflicts
    cleaned_coins = {}
    period_conflicts = 0
    
    for coin_id, data in raw_groups.items():
        clean_sides = {}
        periods_seen = set()
        
        for side, candidates in data['sides'].items():
            # 1. Pick Best Candidate for this side
            # Sort by quality desc, then by manifest_idx (stable tie-break)
            best = max(candidates, key=lambda x: (x['quality'], x['manifest_idx']))
            clean_sides[side] = best
            
            # Collect periods
            for c in candidates:
                if c['period']: periods_seen.add(c['period'])
        
        # 2. Resolve Period (Strictly Deterministic)
        final_period = None
        sorted_periods = sorted(list(periods_seen))
        
        if len(sorted_periods) == 1:
            final_period = sorted_periods[0]
        elif len(sorted_periods) > 1:
            period_conflicts += 1
            # Priority: Obverse Period > Reverse Period > Alphabetical sort
            if 'obverse' in clean_sides and clean_sides['obverse']['period']:
                final_period = clean_sides['obverse']['period']
            elif 'reverse' in clean_sides and clean_sides['reverse']['period']:
                final_period = clean_sides['reverse']['period']
            else:
                final_period = sorted_periods[0]
                
        cleaned_coins[coin_id] = {
            'sides': clean_sides,
            'period': final_period
        }
    
    if period_conflicts > 0:
        print(f"  ⚠️  Resolved {period_conflicts} coins with conflicting period labels")
        
    return cleaned_coins


def normalize_vector(v):
    """Return unit vector."""
    norm = np.linalg.norm(v)
    return v / (norm + 1e-12)


def export_coin_embeddings(coins: dict, output_dir: Path):
    print("\n📤 Exporting coin-level embeddings...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_coins = []
    complete_pairs = []
    
    for coin_id, data in coins.items():
        sides = data['sides']
        
        # 1. Select embeddings for Average
        sources = []
        if 'obverse' in sides: sources.append(sides['obverse']['embedding'])
        if 'reverse' in sides: sources.append(sides['reverse']['embedding'])
        
        if not sources and 'unknown' in sides:
            sources.append(sides['unknown']['embedding'])
            
        if not sources:
            continue
            
        # Defensive: Normalize sources BEFORE averaging
        sources = [normalize_vector(s) for s in sources]
        
        # Average & Re-Normalize
        avg_emb = np.mean(sources, axis=0)
        avg_emb = normalize_vector(avg_emb)
        
        all_coins.append({
            'coin_identity': coin_id,
            'embedding_avg': avg_emb,
            'period': data['period'],
            'sides_available': list(sides.keys())
        })
        
        # 2. Handle Concatenation (Strict Pairs)
        if 'obverse' in sides and 'reverse' in sides:
            obv = sides['obverse']['embedding']
            rev = sides['reverse']['embedding']
            
            # Normalize SIDES before concat (Critical for balance)
            obv = normalize_vector(obv)
            rev = normalize_vector(rev)
            
            concat = np.concatenate([obv, rev])
            # Normalize result
            concat = normalize_vector(concat)
            
            complete_pairs.append({
                'coin_identity': coin_id,
                'embedding_concat': concat,
                'period': data['period'],
                'obverse_path': sides['obverse']['path'],
                'reverse_path': sides['reverse']['path']
            })

    # Save AVG
    avg_vecs = np.array([c['embedding_avg'] for c in all_coins], dtype=np.float32)
    np.save(output_dir / "coin_embeddings_avg.npy", avg_vecs)
    
    avg_meta = {
        'coin_identities': [c['coin_identity'] for c in all_coins],
        'periods': [c['period'] for c in all_coins],
        'sides_available': [c['sides_available'] for c in all_coins]
    }
    with open(output_dir / "coin_embeddings_avg_meta.json", 'w') as f:
        json.dump(avg_meta, f, indent=2)
        
    # Save CONCAT
    if complete_pairs:
        concat_vecs = np.array([c['embedding_concat'] for c in complete_pairs], dtype=np.float32)
        np.save(output_dir / "coin_embeddings_concat.npy", concat_vecs)
        
        concat_meta = {
            'coin_identities': [c['coin_identity'] for c in complete_pairs],
            'periods': [c['period'] for c in complete_pairs],
            'obverse_paths': [c['obverse_path'] for c in complete_pairs],
            'reverse_paths': [c['reverse_path'] for c in complete_pairs]
        }
        with open(output_dir / "coin_embeddings_concat_meta.json", 'w') as f:
            json.dump(concat_meta, f, indent=2)

    print(f"  ✅ Processed {len(all_coins)} coins ({len(complete_pairs)} complete pairs)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--embeddings-meta', required=True)
    parser.add_argument('--output', default='coin_embeddings')
    args = parser.parse_args()
    
    vectors, paths, confs = load_side_embeddings(args.embeddings, args.embeddings_meta)
    manifest = load_manifest_with_coin_identity(args.manifest)
    path_index = build_path_index(paths)
    
    coins = group_by_coin_identity(manifest, vectors, path_index, confs)
    export_coin_embeddings(coins, args.output)

if __name__ == "__main__":
    main()