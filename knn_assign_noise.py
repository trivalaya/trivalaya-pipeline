#!/usr/bin/env python3
"""
knn_assign_noise.py - Reassign noise points to their nearest cluster based on K-nearest-neighbor voting.

Updates cluster_results.csv in place (with backup) and adds column: assignment_method.
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def main():
    parser = argparse.ArgumentParser(description="Reassign HDBSCAN noise points using KNN voting")
    parser.add_argument('--csv', type=str, default='cluster_output/cluster_results.csv',
                        help='Path to cluster_results.csv')
    parser.add_argument('--embeddings', type=str, default='cluster_output/features.npy',
                        help='Path to features.npy')
    parser.add_argument('--k', type=int, default=5, help='Number of neighbors to consider')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Minimum agreement ratio to assign (e.g., 0.6 means 3/5 neighbors must agree)')
    parser.add_argument('--max-distance', type=float, default=None,
                        help='Optional max cosine distance to consider (default: None)')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    emb_path = Path(args.embeddings)

    df = pd.read_csv(csv_path)
    features = np.load(emb_path)

    # Build KNN index (cosine distance)
    nn = NearestNeighbors(n_neighbors=args.k + 1, metric='cosine')
    nn.fit(features)

    # Track which points were originally clustered
    original_clustered_mask = df['cluster_id'] >= 0

    # Ensure assignment_method column exists
    if 'assignment_method' not in df.columns:
        df['assignment_method'] = ''

    # Set assignment_method for original cluster members
    df.loc[original_clustered_mask, 'assignment_method'] = 'clustered'

    # Default all original noise points to 'noise' (may be overwritten if reassigned)
    df.loc[~original_clustered_mask, 'assignment_method'] = 'noise'

    noise_indices = df.index[df['cluster_id'] == -1].tolist()

    reassigned = 0
    remain_noise = 0

    for idx in noise_indices:
        # Query K+1 neighbors (first is self)
        distances, indices = nn.kneighbors(features[idx].reshape(1, -1), n_neighbors=args.k + 1, return_distance=True)
        distances = distances[0].tolist()
        indices = indices[0].tolist()

        # Drop self (first neighbor)
        distances = distances[1:]
        indices = indices[1:]

        # Optional max distance filter
        if args.max_distance is not None:
            filtered = [(i2, d2) for i2, d2 in zip(indices, distances) if d2 <= args.max_distance]
        else:
            filtered = list(zip(indices, distances))

        # Filter neighbors to only those with cluster_id >= 0
        clustered_neighbors = [i2 for i2, _ in filtered if int(df.loc[i2, 'cluster_id']) >= 0]

        # If fewer than K/2 neighbors have clusters, mark as noise, skip
        if len(clustered_neighbors) < (args.k / 2):
            df.at[idx, 'cluster_id'] = -1
            df.at[idx, 'assignment_method'] = 'noise'
            remain_noise += 1
            continue

        # Count cluster votes among clustered neighbors
        votes = {}
        for ni in clustered_neighbors:
            cid = int(df.loc[ni, 'cluster_id'])
            votes[cid] = votes.get(cid, 0) + 1

        winning_cluster = max(votes.items(), key=lambda x: x[1])[0]
        winning_votes = votes[winning_cluster]

        # If top cluster has >= threshold agreement
        if (winning_votes / args.k) >= args.threshold:
            df.at[idx, 'cluster_id'] = int(winning_cluster)
            df.at[idx, 'assignment_method'] = 'knn_assigned'
            reassigned += 1
        else:
            df.at[idx, 'cluster_id'] = -1
            df.at[idx, 'assignment_method'] = 'noise'
            remain_noise += 1

    # Backup original CSV, save updated CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = csv_path.with_suffix(csv_path.suffix + f".bak_{ts}")
    shutil.copy2(csv_path, backup_path)
    df.to_csv(csv_path, index=False)

    print(f"{reassigned} noise points reassigned, {remain_noise} remain noise")


if __name__ == "__main__":
    main()
