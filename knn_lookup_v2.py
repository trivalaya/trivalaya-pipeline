#!/usr/bin/env python3
"""
knn_lookup_v2.py - Precompute KNN neighbor index.
FINAL VERSION: Safe NaN handling and alignment checks.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def normalize_features(feats):
    """Returns L2 normalized copy of features."""
    print("  ⚖️  Normalizing features for Cosine similarity...")
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / (norms + 1e-12)

def filter_by_side(features: np.ndarray, df: pd.DataFrame, 
                   manifest_path: str, target_side: str) -> tuple:
    """Filter embeddings to only include specific side (obverse or reverse)."""
    
    df = df.copy()
    
    if 'side' not in df.columns:
        if not manifest_path:
             raise ValueError("CSV missing 'side' column and no manifest provided.")
             
        print(f"  Mapping sides via image_path from {manifest_path}...")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        path_map = {
            str(r.get('image_path', '')).replace("\\", "/").strip().lstrip("./"): r.get('side', 'unknown')
            for r in manifest
        }
        
        def get_side(p):
            norm = str(p).replace("\\", "/").strip().lstrip("./")
            return path_map.get(norm, 'unknown')
            
        df['side'] = df['image_path'].apply(get_side)

    mask = df['side'] == target_side
    filtered_features = features[mask]
    filtered_df = df[mask].reset_index(drop=True)
    
    print(f"  Filtered to {len(filtered_df)} {target_side} images")
    return filtered_features, filtered_df

def build_knn_index(features: np.ndarray, identifiers: list, meta_lookup: callable, k: int, output_path: str):
    """Generic KNN builder."""
    print(f"\n🔍 Building KNN index (k={k})...")
    
    features = normalize_features(features)
    
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(features)), metric='cosine')
    nn.fit(features)
    
    knn_index = {}
    
    for i in range(len(features)):
        query_id = identifiers[i]
        
        dists, indices = nn.kneighbors(
            features[i].reshape(1, -1), 
            n_neighbors=min(k + 1, len(features)),
            return_distance=True
        )
        
        dists = dists[0].tolist()[1:]
        indices = indices[0].tolist()[1:]
        
        neighbors = []
        for ni, d in zip(indices, dists):
            n_data = meta_lookup(ni)
            n_data['distance'] = float(d)
            neighbors.append(n_data)
            
        knn_index[str(query_id)] = {"neighbors": neighbors}
    
    with open(output_path, 'w') as f:
        json.dump(knn_index, f, indent=2)
    print(f"✅ Wrote {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--csv', help='For side granularity')
    parser.add_argument('--meta', help='For coin granularity')
    parser.add_argument('--manifest', help='For side filtering')
    parser.add_argument('--granularity', choices=['side', 'coin', 'obverse', 'reverse'], default='side')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--output', default='knn_index.json')
    args = parser.parse_args()
    
    features = np.load(args.embeddings)
    
    if args.granularity == 'coin':
        with open(args.meta, 'r') as f:
            meta = json.load(f)
        ids = meta['coin_identities']
        
        def lookup_coin(idx):
            return {
                "coin_identity": ids[idx],
                "period": meta['periods'][idx]
            }
        build_knn_index(features, ids, lookup_coin, args.k, args.output)
        
    else: # side, obverse, reverse
        df = pd.read_csv(args.csv)
        
        # FIX: Alignment Assertion
        if len(df) != len(features):
            if args.granularity in ['obverse', 'reverse']:
                 # We will filter anyway, but initial mismatch is suspicious
                 print(f"⚠️ Warning: Input CSV ({len(df)}) and Embeddings ({len(features)}) length mismatch.")
            else:
                 raise ValueError(f"Length mismatch: CSV has {len(df)} rows, Embeddings has {len(features)} rows.")

        if args.granularity in ['obverse', 'reverse']:
            features, df = filter_by_side(features, df, args.manifest, args.granularity)
            
        ids = df['id'].tolist()
        def lookup_side(idx):
            row = df.iloc[idx]
            # FIX: Safe NaN handling for cluster_id
            cid = row.get('cluster_id', -1)
            cid = -1 if pd.isna(cid) else int(cid)
            
            return {
                "id": int(row['id']),
                "cluster_id": cid,
                "period": str(row.get('period', 'unknown'))
            }
        build_knn_index(features, ids, lookup_side, args.k, args.output)

if __name__ == "__main__":
    main()