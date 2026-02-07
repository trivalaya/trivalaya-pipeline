# Scaling Plan: 23k → 100k Coins (2026-02)

## Current State
- 23,329 coins with CLIP embeddings + enriched metadata
- CPU-only server
- app2.py v0.7.0 with 512-d search features
- ~5-10% of crops have edge clipping (margin fix committed, not yet re-processed)

---

## Phase 1: Clean up current 23k (CPU, immediate)
- [ ] Run `derive_search_embeddings.py` — unblocks app2.py launch
- [ ] Commit `two_coin_resolver.py` margin fix in trivalaya-vision repo
- [ ] Re-crop 23k coins with new 0.12 margin — overnight batch, CPU fine, bottleneck is Spaces I/O

### Phase 1 → 2 integrity checks
- [ ] `features.npy` shape[0] == enriched CSV row count (no orphan embeddings)
- [ ] No duplicate `coin_id` values in enriched CSV
- [ ] Embedding L2 norms within [0.99, 1.01] (unit-length sanity)
- [ ] `search_features_512.npy` shape matches `features.npy` shape[0]

---

## Phase 2: Scale scraping + vision to 100k (CPU)
- [ ] Scrape new auction houses / sales — pure I/O, CPU fine
- [ ] Run vision pipeline on new lots — OpenCV, CPU fine. ~0.5s/image, 77k new lots ~11 hours. Parallelizable across cores
- [ ] Enrich metadata — enrich_cluster_results.py is pandas joins, trivial at 100k

### Phase 2 → 3 integrity checks
- [ ] Enriched CSV covers all new coin_ids from scrape + vision
- [ ] No duplicate coin_ids across old + new batches
- [ ] Required columns non-null: obv_path, rev_path, parser_period, denomination
- [ ] Crop files exist in Spaces for every obv_path/rev_path referenced

### Data-freeze boundary
Before entering Phase 3, **freeze the scrape snapshot**:
- [ ] Lock the scrape manifest — no new lots ingested past this point
- [ ] Record exact row count and coin_id range in a freeze manifest (`freeze_v2.json`)
- [ ] Validate period-label coverage: ≥90% of coins have a parser_period value
- [ ] Validate denomination coverage: ≥80% non-null
- [ ] Confirm re-crop is complete (0.12 margin) for entire frozen set
- [ ] Commit enriched CSV + freeze manifest to version control

All embedding, clustering, and training in subsequent phases operate on this frozen snapshot. New scrapes queue into the next cycle.

---

## Phase 3: Re-embed everything (GPU boundary)

### Go/no-go checklist (must pass before GPU spin-up)
- [ ] Re-crop complete for frozen set — spot-check 50 random crops visually
- [ ] Enriched CSV QA report generated (row count, null rates, period distribution)
- [ ] `derive_search_embeddings.py` validated on current 23k — search results manually reviewed
- [ ] Dry-run embedding on 5k sample: output shape correct, norms ~1.0, no NaN/Inf
- [ ] GPU script (`embed_full_dataset.py`) supports resume from checkpoint (handles OOM / preemption)
- [ ] Spaces write permissions confirmed for output prefix

### Embedding run
- [ ] CLIP embedding of 100k pairs — CPU hits the wall here. On CPU: ViT-B-32 ~200ms/image, 200k sides = ~11 hours. On DO GPU droplet (A10/H100): ~30 minutes at batch_size=256. Worth $2-4/hr for a one-shot job.

### Incremental embedding path
Embeddings are stored as immutable versioned snapshots to avoid full re-embeds on every data addition:

```
embeddings/
  index_v1/          # 23k frozen snapshot
    features.npy
    metadata.csv
    freeze_manifest.json
  index_v2/          # 100k frozen snapshot
    features.npy
    metadata.csv
    freeze_manifest.json
  delta_v3/          # new coins since v2 freeze
    features.npy
    metadata.csv
```

- Each `index_vN/` is immutable once written — never modified after creation
- Between freezes, new coins go into `delta_vN+1/` (small append-only batches, CPU is fine)
- At serving time, concatenate `index_vN/` + `delta_vN+1/` in memory
- Periodic merge: when delta exceeds ~10% of index size, freeze a new `index_vN+1/`

### Phase 3 → 4 integrity checks
- [ ] `features.npy` shape[0] == frozen CSV row count
- [ ] All embedding norms within [0.99, 1.01]
- [ ] No NaN or Inf values in feature matrix
- [ ] Round-trip test: embed 10 known coins, verify cosine similarity > 0.95 to themselves

---

## Phase 4: Clustering (optional for serving — async quality layer)

**Clustering is NOT required to ship search.** Deploy visual search (Phase 6) as soon as embeddings are ready. Run clustering asynchronously as a quality and exploration layer.

- [ ] UMAP + HDBSCAN on 100k x 512 — UMAP O(n log n) but heavy constant. At 23k: 10-20 min. At 100k: 1-3 hours on CPU. Workable overnight. GPU with cuML UMAP cuts to minutes if iterating on hyperparameters (n_neighbors, min_cluster_size).
- [ ] Cluster assignments feed into the UI as browseable groups and quality audit (misclassified coins surface as cluster outliers)
- [ ] Clustering does not block search serving or classifier training

---

## Phase 5: Retrain period classifier (GPU required)
- [ ] MobileNetV2 fine-tuning on 100k pairs — needs GPU. A10 handles in 1-2 hours. Training on CPU at 100k is impractical (days).

### Success criteria
Training is not "done" until these thresholds are met on the held-out test split:

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Macro F1 | ≥ 0.80 | Weighted equally across all period classes |
| Per-class recall | ≥ 0.60 for every class with ≥50 samples | No silent class collapse |
| Calibration | ECE ≤ 0.10 | Confidence scores should be meaningful for reranking |
| Confusion hotspots | Document top-5 confused class pairs | e.g., "Republican" vs "Imperatorial" — drives label cleanup |

- [ ] Generate full confusion matrix and save to `artifacts/`
- [ ] Review top confusion pairs — determine if label noise or genuine ambiguity
- [ ] If per-class recall floor is missed, investigate: class size, label quality, visual similarity
- [ ] Compare v2 model against v1 on the 23k test set (regression check)

---

## Phase 6: Rebuild search index + deploy
- [ ] Re-run `derive_search_embeddings.py` — seconds, CPU
- [ ] Redeploy app2.py with new index

### Serving latency note
At 23k, brute-force 512-d matrix multiply is ~2ms — negligible. At 100k this grows but remains fast for the dot-product itself. However, **end-to-end query p95 is higher** because it includes:
- CLIP encode of query image: ~15-30ms on CPU (ViT-B-32)
- Dot-product search over 100k x 512: ~5-10ms
- Metadata join + result serialization: ~2-5ms
- **Estimated p95 at 100k: ~30-50ms on CPU** — well within interactive budget

Future optimizations (not needed at 100k, consider at 500k+):
- **Prefilter by period**: partition index by parser_period, search only the relevant partition. Cuts search space 5-10x.
- **float16 index**: halves memory footprint and improves cache locality. Negligible accuracy loss for cosine similarity at 512-d.
- **FAISS IVF index**: approximate nearest neighbors if brute-force exceeds latency budget.

---

## CPU vs GPU Summary

| Stage              | 23k (now) | 100k       | GPU needed?                      |
|--------------------|-----------|------------|----------------------------------|
| Scrape + vision    | hours     | overnight  | No                               |
| CLIP embedding     | ~3hr      | ~11hr      | **Yes** — saves 10x time         |
| UMAP + HDBSCAN     | 20min     | 1-3hr      | Optional — CPU works overnight   |
| Training           | 30min     | days       | **Yes** — impractical without    |
| Search serving     | ~2ms/qry  | ~40ms/qry  | No — includes CLIP encode        |

## GPU Strategy
Spin up a DO GPU droplet for a single day to run embedding (Phase 3) + training (Phase 5), then tear it down. Everything else runs on current CPU box.
