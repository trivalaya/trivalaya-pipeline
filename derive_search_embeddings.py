"""
Derive 512-d search vectors from 1024-d paired CLIP features.

The paired features (obv || rev, each 512-d) were L2-normalized per-side,
concatenated, then L2-normalized as a 1024-d vector. Splitting recovers the
halves but they are no longer unit-length, so we re-normalize before averaging.

Input:  cluster_output_clip_spaces/features.npy  (N x 1024)
Output: cluster_output_clip_spaces/search_features_512.npy  (N x 512)
"""

import numpy as np
from pathlib import Path

SRC = Path("cluster_output_clip_spaces/features.npy")
DST = Path("cluster_output_clip_spaces/search_features_512.npy")


def main():
    features = np.load(SRC)
    assert features.shape[1] == 1024, f"Expected 1024-d, got {features.shape[1]}-d"
    print(f"Loaded {features.shape[0]} x {features.shape[1]} features from {SRC}")

    obv = features[:, :512].copy()
    rev = features[:, 512:].copy()

    # Re-normalize each half to unit length
    obv /= np.linalg.norm(obv, axis=1, keepdims=True) + 1e-12
    rev /= np.linalg.norm(rev, axis=1, keepdims=True) + 1e-12

    # Average and L2-normalize
    avg = (obv + rev) / 2.0
    avg /= np.linalg.norm(avg, axis=1, keepdims=True) + 1e-12

    avg = avg.astype(np.float32)
    np.save(DST, avg)
    print(f"Saved {avg.shape[0]} x {avg.shape[1]} search features to {DST}")


if __name__ == "__main__":
    main()
