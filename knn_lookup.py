#!/usr/bin/env python3
"""
knn_lookup.py - Precompute KNN neighbor index as JSON for a static viewer.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def main():
    parser = argparse.ArgumentParser(description="Precompute KNN index JSON for coin embeddings")
    parser.add_argument('--embeddings', type=str, required=True, help='Path to features.npy')
    parser.add_argument('--csv', type=str, required=True, help='Path to cluster_results.csv')
    parser.add_argument('--k', type=int, default=10, help='Neighbors to precompute')
    parser.add_argument('--output', type=str, default='cluster_output/knn_index.json', help='Output JSON path')
    args = parser.parse_args()

    features = np.load(Path(args.embeddings))
    df = pd.read_csv(Path(args.csv))

    nn = NearestNeighbors(n_neighbors=args.k + 1, metric='cosine')
    nn.fit(features)

    knn_index = {}

    for i in range(len(df)):
        coin_id = int(df.loc[i, 'id'])
        distances, indices = nn.kneighbors(features[i].reshape(1, -1), n_neighbors=args.k + 1, return_distance=True)
        distances = distances[0].tolist()
        indices = indices[0].tolist()

        # Drop self (first neighbor)
        distances = distances[1:]
        indices = indices[1:]

        neighbors = []
        for ni, d in zip(indices, distances):
            nid = int(df.loc[ni, 'id'])
            cluster_id = int(df.loc[ni, 'cluster_id'])
            if 'parser_period' in df.columns:
                period = df.loc[ni, 'parser_period']
            elif 'period' in df.columns:
                period = df.loc[ni, 'period']
            else:
                period = 'unknown'
            neighbors.append({
                "id": nid,
                "distance": float(d),
                "cluster_id": cluster_id,
                "period": str(period) if period is not None else "unknown"
            })

        knn_index[str(coin_id)] = {"neighbors": neighbors}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(knn_index, f, indent=2)

    print(f"Wrote {out_path} ({len(knn_index)} coins)")


if __name__ == "__main__":
    main()
