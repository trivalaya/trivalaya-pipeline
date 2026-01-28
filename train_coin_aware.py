#!/usr/bin/env python3
"""
train_final_coin_aware.py - Trivalaya coin training (single-side OR coin-aware obverse+reverse pairs)

This is a *merged* "gold master" training script:

- Keeps the deterministic / production-grade behavior and manifest-aligned embedding export from train_final.py
- Adds coin-aware training that can use true obv+rev pairs (when present in the manifest)
- Supports BOTH manifest formats in one file:
    A) Single-side records: {"image_path": "...", "period": "...", ...}
    B) Pair records: {"coin_identity": "...", "period": "...", "obverse": {"image_path": "..."}, "reverse": {"image_path": "..."}}

USAGE
-----
Single-side (your current pipeline):
    python train_final_coin_aware.py --manifest trivalaya_data/03_ml_ready/dataset/train_manifest.json --mode sides

Coin-aware pairs (requires pair-form manifest):
    python train_final_coin_aware.py --manifest trivalaya_data/03_ml_ready/dataset/train_manifest_pairs.json --mode pairs

Auto mode (uses pairs if present, else sides):
    python train_final_coin_aware.py --manifest ... --mode auto

Determinism:
    python train_final_coin_aware.py --manifest ... --mode sides --strict-determinism

Outputs
-------
Sides:
    trivalaya_model_sides.pth
    trivalaya_model_sides_meta.json
    trivalaya_embeddings_sides.npy
    trivalaya_embeddings_sides_meta.json

Pairs:
    trivalaya_model_pairs.pth
    trivalaya_model_pairs_meta.json
    trivalaya_embeddings_pairs.npy
    trivalaya_embeddings_pairs_meta.json
"""

# --- ENVIRONMENT SETUP (must be before torch import when strict determinism requested) ---
import os
import sys

DEFAULT_SEED = 42
if "--strict-determinism" in sys.argv:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(DEFAULT_SEED)

# --- IMPORTS ---
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from PIL import Image


# -------------------------
# Determinism / Reproducibility
# -------------------------
def set_determinism(seed: int = DEFAULT_SEED, strict: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = strict
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if strict:
        # Throw if any op is nondeterministic
        torch.use_deterministic_algorithms(True)
        print("🔒 Strict determinism ENABLED")
    else:
        print("⚠️  Strict determinism DISABLED")


def seed_worker(worker_id: int) -> None:
    # Ensures deterministic transforms across DataLoader workers
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 16
    epochs: int = 20
    lr: float = 5e-4
    max_weight_cap: float = 10.0
    image_size: int = 128
    aug_rotation: int = 15
    aug_scale: Tuple[float, float] = (0.9, 1.1)
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Pair model specific
    fusion_hidden: int = 512
    dropout: float = 0.3


# -------------------------
# Manifest Loading
# -------------------------
def load_manifest(manifest_path: Path) -> Tuple[List[Dict[str, Any]], int]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"❌ Manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        raw = json.load(f)

    # Stamp manifest index for alignment downstream
    for i, rec in enumerate(raw):
        rec["_manifest_index"] = i

    return raw, len(raw)


def infer_manifest_kind(records: List[Dict[str, Any]]) -> str:
    """
    Returns: 'pairs' if *any* record looks like a pair record, else 'sides'.

    Supported pair schemas:
      A) {'obverse': {'image_path': ...}, 'reverse': {'image_path': ...}}
      B) {'obv_path': ..., 'rev_path': ...}
      C) {'obverse_path': ..., 'reverse_path': ...}
    """
    for r in records:
        # A) nested objects
        if isinstance(r.get("obverse"), dict) and isinstance(r.get("reverse"), dict):
            if r["obverse"].get("image_path") and r["reverse"].get("image_path"):
                return "pairs"
        # B/C) flat keys
        if r.get("obv_path") and r.get("rev_path"):
            return "pairs"
        if r.get("obverse_path") and r.get("reverse_path"):
            return "pairs"
    return "sides"


def get_pair_paths(r: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Return (obv_path, rev_path) for any supported pair schema."""
    if isinstance(r.get("obverse"), dict) and isinstance(r.get("reverse"), dict):
        obv = r["obverse"].get("image_path")
        rev = r["reverse"].get("image_path")
        return obv, rev
    obv = r.get("obv_path") or r.get("obverse_path")
    rev = r.get("rev_path") or r.get("reverse_path")
    return obv, rev


def record_has_complete_pair(r: Dict[str, Any]) -> bool:
    """True if the record has usable obv+rev paths and (if present) has_obv/has_rev flags are truthy."""
    obv, rev = get_pair_paths(r)
    if not (obv and rev):
        return False

    # Respect flags if they exist
    def _flag_truthy(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "y", "t"}
        return bool(v)

    if "has_obv" in r and not _flag_truthy(r.get("has_obv")):
        return False
    if "has_rev" in r and not _flag_truthy(r.get("has_rev")):
        return False
    return True


# -------------------------
# Common label mapping + weights
# -------------------------
def build_period_mapping(records: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, int]]:
    periods = sorted({r["period"] for r in records if r.get("period")})
    period_to_idx = {p: i for i, p in enumerate(periods)}
    return periods, period_to_idx


def compute_class_weights(train_period_counts: Dict[str, int], periods: List[str], max_cap: float) -> torch.Tensor:
    total_train = sum(train_period_counts.get(p, 0) for p in periods)
    weights: List[float] = []
    for p in periods:
        cnt = train_period_counts.get(p, 0)
        if cnt <= 0:
            w = 0.0
        else:
            raw = total_train / (len(periods) * cnt)
            w = min(raw, max_cap)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


# -------------------------
# Transforms (coin-safe)
# -------------------------
def make_transforms(cfg: TrainConfig) -> Tuple[transforms.Compose, transforms.Compose]:
    train_t = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomAffine(
            degrees=cfg.aug_rotation,
            translate=(0.05, 0.05),
            scale=cfg.aug_scale
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_t = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_t, val_t


# -------------------------
# Datasets + Collate
# -------------------------
class SideDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], transform, period_to_idx: Dict[str, int], skipped_files: set):
        self.records = records
        self.transform = transform
        self.period_to_idx = period_to_idx
        self.skipped_files = skipped_files

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        path = r.get("image_path")
        try:
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")

            with Image.open(path) as im:
                img = im.convert("RGB")

            y = self.period_to_idx[r["period"]]
            mi = r["_manifest_index"]
            return self.transform(img), torch.tensor(y, dtype=torch.long), mi, path
        except Exception:
            if path:
                self.skipped_files.add(path)
            return None


class PairDataset(Dataset):
    def __init__(self, pairs: List[Dict[str, Any]], transform, period_to_idx: Dict[str, int], skipped_files: set):
        self.pairs = pairs
        self.transform = transform
        self.period_to_idx = period_to_idx
        self.skipped_files = skipped_files

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        r = self.pairs[idx]
        obv_path = (
            r.get("obverse", {}).get("image_path")
            or r.get("obv_path")
            or r.get("obverse_path")
        )
        rev_path = (
            r.get("reverse", {}).get("image_path")
            or r.get("rev_path")
            or r.get("reverse_path")
        )
        try:
            if not obv_path or not os.path.exists(obv_path):
                raise FileNotFoundError(f"Missing obverse: {obv_path}")
            if not rev_path or not os.path.exists(rev_path):
                raise FileNotFoundError(f"Missing reverse: {rev_path}")

            with Image.open(obv_path) as im:
                obv = im.convert("RGB")
            with Image.open(rev_path) as im:
                rev = im.convert("RGB")

            y = self.period_to_idx[r["period"]]
            mi = r["_manifest_index"]
            coin_id = (
            r.get("coin_identity")
            or r.get("coin_id")
            or f"pair_{mi}"
        )
            return self.transform(obv), self.transform(rev), torch.tensor(y, dtype=torch.long), mi, coin_id, obv_path, rev_path
        except Exception:
            if obv_path:
                self.skipped_files.add(obv_path)
            if rev_path:
                self.skipped_files.add(rev_path)
            return None


def collate_skip_none_sides(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs, labels, manifest_idxs, paths = zip(*batch)
    return (
        torch.stack(imgs),
        torch.stack(labels),
        torch.tensor(manifest_idxs, dtype=torch.long),
        list(paths),
    )


def collate_skip_none_pairs(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    obvs, revs, labels, manifest_idxs, coin_ids, obv_paths, rev_paths = zip(*batch)
    return (
        torch.stack(obvs),
        torch.stack(revs),
        torch.stack(labels),
        torch.tensor(manifest_idxs, dtype=torch.long),
        list(coin_ids),
        list(obv_paths),
        list(rev_paths),
    )


# -------------------------
# Models
# -------------------------
class SideClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.backbone.classifier[1] = nn.Linear(self.backbone.last_channel, num_classes)

    def forward(self, x):
        return self.backbone(x)

    @property
    def feat_dim(self) -> int:
        return self.backbone.last_channel


class CoinPairClassifier(nn.Module):
    def __init__(self, num_classes: int, fusion_hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone.last_channel
        self.fusion = nn.Sequential(
            nn.Linear(self.feat_dim * 2, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, obv, rev):
        f1 = self.pool(self.features(obv)).flatten(1)
        f2 = self.pool(self.features(rev)).flatten(1)
        return self.fusion(torch.cat([f1, f2], dim=1))


# -------------------------
# Splits
# -------------------------
def split_sides(records: List[Dict[str, Any]], seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """
    Deterministic per-class split, same inclusive strategy you were using.
    Returns train_records, val_records, train_counts_by_period
    """
    rng = random.Random(seed)
    by_period = defaultdict(list)
    for r in records:
        if r.get("period") and r.get("image_path"):
            by_period[r["period"]].append(r)

    train, val = [], []
    train_counts = defaultdict(int)

    for period in sorted(by_period.keys()):
        items = by_period[period]
        rng.shuffle(items)

        if len(items) == 1:
            train.extend(items)
            train_counts[period] += 1
        else:
            n_val = max(1, int(len(items) * 0.2))
            val.extend(items[:n_val])
            train.extend(items[n_val:])
            train_counts[period] += (len(items) - n_val)

    return train, val, train_counts


def split_pairs(pairs: List[Dict[str, Any]], seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """
    Identity-aware split: we split by coin_identity (not by image) to avoid leakage.
    We do it within each period, using ~20% of identities for validation.
    Returns train_pairs, val_pairs, train_counts_by_period (counting pairs).
    """
    rng = random.Random(seed)

    by_period = defaultdict(list)
    for r in pairs:
        if r.get("period") and r.get("coin_identity"):
            by_period[r["period"]].append(r)

    train, val = [], []
    train_counts = defaultdict(int)

    for period in sorted(by_period.keys()):
        items = by_period[period]

        # group by identity
        by_id = defaultdict(list)
        for r in items:
            by_id[r["coin_identity"]].append(r)

        ids = list(by_id.keys())
        rng.shuffle(ids)

        if len(ids) == 1:
            chosen_val = set()
        else:
            n_val_ids = max(1, int(len(ids) * 0.2))
            chosen_val = set(ids[:n_val_ids])

        for cid, recs in by_id.items():
            if cid in chosen_val:
                val.extend(recs)
            else:
                train.extend(recs)
                train_counts[period] += len(recs)

    return train, val, train_counts


# -------------------------
# Training helpers
# -------------------------
def build_loader(dataset: Dataset, cfg: TrainConfig, shuffle: bool, collate_fn):
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_memory and torch.cuda.is_available()),
        worker_init_fn=seed_worker,
        generator=g,
    )

    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = cfg.persistent_workers
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor

    return DataLoader(dataset, **loader_kwargs)


def train_loop_sides(
    cfg: TrainConfig,
    periods: List[str],
    period_to_idx: Dict[str, int],
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    class_weights: torch.Tensor,
    out_prefix: str,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skipped_files: set = set()

    train_t, val_t = make_transforms(cfg)

    train_ds = SideDataset(train_records, train_t, period_to_idx, skipped_files)
    val_ds = SideDataset(val_records, val_t, period_to_idx, skipped_files)

    train_loader = build_loader(train_ds, cfg, shuffle=True, collate_fn=collate_skip_none_sides)
    val_loader = build_loader(val_ds, cfg, shuffle=False, collate_fn=collate_skip_none_sides)

    model = SideClassifier(num_classes=len(periods)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = 0.0
    model_path = f"{out_prefix}_sides.pth"

    print(f"\n🚀 [SIDES] Training on {device} | train={len(train_records)} val={len(val_records)} classes={len(periods)}")
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_seen = 0
        steps = 0

        for batch in train_loader:
            if batch is None:
                continue
            imgs, labels, _, _ = batch
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            train_correct += int((logits.argmax(1) == labels).sum().item())
            train_seen += int(labels.size(0))
            steps += 1

        # val
        model.eval()
        val_correct = 0
        val_seen = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                imgs, labels, _, _ = batch
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(imgs)
                val_correct += int((logits.argmax(1) == labels).sum().item())
                val_seen += int(labels.size(0))

        train_acc = (train_correct / max(1, train_seen)) * 100.0
        val_acc = (val_correct / max(1, val_seen)) * 100.0
        avg_loss = train_loss / max(1, steps)

        print(f"Epoch {epoch+1:2d} | Loss {avg_loss:.3f} | Train {train_acc:.0f}% | Val {val_acc:.0f}%")

        if val_acc >= best_val:
            best_val = val_acc
            torch.save(model.state_dict(), model_path)

    meta = {
        "mode": "sides",
        "best_val_acc": best_val,
        "period_to_idx": period_to_idx,
        "idx_to_period": periods,
        "config": cfg.__dict__,
        "model_path": model_path,
        "skipped_count": len(skipped_files),
        "skipped_log": f"{out_prefix}_skipped_sides.txt",
        "backbone": "mobilenet_v2",
        "feature_dim": model.feat_dim,
    }

    if skipped_files:
        with open(meta["skipped_log"], "w") as f:
            for p in sorted(skipped_files):
                f.write(p + "\n")

    with open(f"{out_prefix}_sides_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"💾 Saved model: {model_path}")
    print(f"💾 Saved meta : {out_prefix}_sides_meta.json")

    return {"model": model, "val_transform": val_t, "meta": meta, "skipped_files": skipped_files}


def train_loop_pairs(
    cfg: TrainConfig,
    periods: List[str],
    period_to_idx: Dict[str, int],
    train_pairs: List[Dict[str, Any]],
    val_pairs: List[Dict[str, Any]],
    class_weights: torch.Tensor,
    out_prefix: str,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skipped_files: set = set()

    train_t, val_t = make_transforms(cfg)

    train_ds = PairDataset(train_pairs, train_t, period_to_idx, skipped_files)
    val_ds = PairDataset(val_pairs, val_t, period_to_idx, skipped_files)

    train_loader = build_loader(train_ds, cfg, shuffle=True, collate_fn=collate_skip_none_pairs)
    val_loader = build_loader(val_ds, cfg, shuffle=False, collate_fn=collate_skip_none_pairs)

    model = CoinPairClassifier(num_classes=len(periods), fusion_hidden=cfg.fusion_hidden, dropout=cfg.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = 0.0
    model_path = f"{out_prefix}_pairs.pth"

    print(f"\n🚀 [PAIRS] Training on {device} | train={len(train_pairs)} val={len(val_pairs)} classes={len(periods)}")
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_seen = 0
        steps = 0

        for batch in train_loader:
            if batch is None:
                continue
            obv, rev, labels, _, _, _, _ = batch
            obv = obv.to(device, non_blocking=True)
            rev = rev.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(obv, rev)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            train_correct += int((logits.argmax(1) == labels).sum().item())
            train_seen += int(labels.size(0))
            steps += 1

        # val
        model.eval()
        val_correct = 0
        val_seen = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                obv, rev, labels, _, _, _, _ = batch
                obv = obv.to(device, non_blocking=True)
                rev = rev.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(obv, rev)
                val_correct += int((logits.argmax(1) == labels).sum().item())
                val_seen += int(labels.size(0))

        train_acc = (train_correct / max(1, train_seen)) * 100.0
        val_acc = (val_correct / max(1, val_seen)) * 100.0
        avg_loss = train_loss / max(1, steps)

        print(f"Epoch {epoch+1:2d} | Loss {avg_loss:.3f} | Train {train_acc:.0f}% | Val {val_acc:.0f}%")

        if val_acc >= best_val:
            best_val = val_acc
            torch.save(model.state_dict(), model_path)

    meta = {
        "mode": "pairs",
        "best_val_acc": best_val,
        "period_to_idx": period_to_idx,
        "idx_to_period": periods,
        "config": cfg.__dict__,
        "model_path": model_path,
        "skipped_count": len(skipped_files),
        "skipped_log": f"{out_prefix}_skipped_pairs.txt",
        "backbone": "mobilenet_v2",
        "feature_dim_single": 1280,
        "feature_dim_pair_concat": 2560,
    }

    if skipped_files:
        with open(meta["skipped_log"], "w") as f:
            for p in sorted(skipped_files):
                f.write(p + "\n")

    with open(f"{out_prefix}_pairs_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"💾 Saved model: {model_path}")
    print(f"💾 Saved meta : {out_prefix}_pairs_meta.json")

    return {"model": model, "val_transform": val_t, "meta": meta, "skipped_files": skipped_files}


# -------------------------
# Embedding export (manifest-aligned)
# -------------------------
def export_embeddings_sides(
    cfg: TrainConfig,
    records_all: List[Dict[str, Any]],
    periods: List[str],
    period_to_idx: Dict[str, int],
    trained_model: SideClassifier,
    val_transform,
    manifest_length: int,
    out_prefix: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.eval()

    # Feature extractor head (same as train_final.py style)
    feature_extractor = torch.nn.Sequential(
        trained_model.backbone.features,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
    ).to(device).eval()

    # Keep only valid, preserve manifest order
    full = [r for r in records_all if r.get("period") and r.get("image_path")]
    skipped_files: set = set()
    ds = SideDataset(full, val_transform, period_to_idx, skipped_files)

    # O(1) lookup manifest index -> path
    manifest_to_path = {r["_manifest_index"]: r["image_path"] for r in full}

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_skip_none_sides,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_memory and torch.cuda.is_available()),
        worker_init_fn=seed_worker,
    )

    embeddings = []
    labels_list: List[str] = []
    paths_list: List[str] = []
    manifest_indices: List[int] = []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            imgs, labels, m_idxs, _paths = batch
            if imgs.numel() == 0:
                continue
            imgs = imgs.to(device, non_blocking=True)
            feats = feature_extractor(imgs)  # (B, 1280)
            feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-12)  # L2

            embeddings.append(feats.cpu().numpy())
            lbls = labels.cpu().numpy().tolist()
            labels_list.extend([periods[int(l)] for l in lbls])

            idxs = m_idxs.cpu().numpy().tolist()
            manifest_indices.extend(idxs)
            paths_list.extend([manifest_to_path[i] for i in idxs])

    out_vec = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 1280), dtype=np.float32)
    np.save(f"{out_prefix}_embeddings_sides.npy", out_vec)

    with open(f"{out_prefix}_embeddings_sides_meta.json", "w") as f:
        json.dump(
            {
                "mode": "sides",
                "paths": paths_list,
                "labels": labels_list,
                "manifest_indices": manifest_indices,
                "manifest_length": manifest_length,
                "period_to_idx": period_to_idx,
                "idx_to_period": periods,
                "feature_dim": int(out_vec.shape[1]) if out_vec.ndim == 2 else 0,
                "normalized": True,
                "count": int(out_vec.shape[0]),
            },
            f,
            indent=2,
        )

    print(f"\n🧬 [SIDES] Saved {out_vec.shape[0]} embeddings → {out_prefix}_embeddings_sides.npy")
    print(f"    Manifest coverage: {out_vec.shape[0]}/{manifest_length}")


def export_embeddings_pairs(
    cfg: TrainConfig,
    records_all: List[Dict[str, Any]],
    periods: List[str],
    period_to_idx: Dict[str, int],
    trained_model: CoinPairClassifier,
    val_transform,
    manifest_length: int,
    out_prefix: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.eval()

    # Use the *shared* feature trunk; export concat(obv_feat, rev_feat) then L2 normalize
    trunk = trained_model.features.to(device).eval()
    pool = nn.AdaptiveAvgPool2d(1).to(device).eval()

    full = [
        r for r in records_all
        if r.get("period") and record_has_complete_pair(r)
    ]
    skipped_files: set = set()
    ds = PairDataset(full, val_transform, period_to_idx, skipped_files)

    # O(1) lookup for paths
    manifest_to_paths = {r["_manifest_index"]: get_pair_paths(r) for r in full}

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_skip_none_pairs,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_memory and torch.cuda.is_available()),
        worker_init_fn=seed_worker,
    )

    embeddings = []
    labels_list: List[str] = []
    obv_paths_list: List[str] = []
    rev_paths_list: List[str] = []
    coin_ids_list: List[str] = []
    manifest_indices: List[int] = []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            obv, rev, labels, m_idxs, coin_ids, _obv_paths, _rev_paths = batch
            if obv.numel() == 0:
                continue
            obv = obv.to(device, non_blocking=True)
            rev = rev.to(device, non_blocking=True)

            f1 = pool(trunk(obv)).flatten(1)  # (B,1280)
            f2 = pool(trunk(rev)).flatten(1)  # (B,1280)
            feats = torch.cat([f1, f2], dim=1)  # (B,2560)
            feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-12)

            embeddings.append(feats.cpu().numpy())
            lbls = labels.cpu().numpy().tolist()
            labels_list.extend([periods[int(l)] for l in lbls])

            idxs = m_idxs.cpu().numpy().tolist()
            manifest_indices.extend(idxs)

            # keep stable manifest-based paths (avoids any loader re-order confusion)
            for i, cid in zip(idxs, coin_ids):
                op, rp = manifest_to_paths[i]
                obv_paths_list.append(op)
                rev_paths_list.append(rp)
                coin_ids_list.append(cid)

    out_vec = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 2560), dtype=np.float32)
    np.save(f"{out_prefix}_embeddings_pairs.npy", out_vec)

    with open(f"{out_prefix}_embeddings_pairs_meta.json", "w") as f:
        json.dump(
            {
                "mode": "pairs",
                "obverse_paths": obv_paths_list,
                "reverse_paths": rev_paths_list,
                "coin_identities": coin_ids_list,
                "labels": labels_list,
                "manifest_indices": manifest_indices,
                "manifest_length": manifest_length,
                "period_to_idx": period_to_idx,
                "idx_to_period": periods,
                "feature_dim": int(out_vec.shape[1]) if out_vec.ndim == 2 else 0,
                "normalized": True,
                "count": int(out_vec.shape[0]),
            },
            f,
            indent=2,
        )

    print(f"\n🧬 [PAIRS] Saved {out_vec.shape[0]} embeddings → {out_prefix}_embeddings_pairs.npy")
    print(f"    Manifest coverage: {out_vec.shape[0]}/{manifest_length}")


# -------------------------
# Main
# -------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to train_manifest.json (sides) or pair-form manifest (pairs).")
    parser.add_argument("--mode", choices=["auto", "sides", "pairs"], default="auto")
    parser.add_argument("--output-prefix", default="trivalaya", help="Prefix for saved artifacts.")
    parser.add_argument("--strict-determinism", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    cfg.seed = DEFAULT_SEED
    set_determinism(cfg.seed, strict=args.strict_determinism)

    manifest_path = Path(args.manifest)
    records_all, manifest_length = load_manifest(manifest_path)

    # Filter valid records (must have period)
    valid = [r for r in records_all if r.get("period")]
    if not valid:
        raise RuntimeError("No valid records with 'period' found in manifest.")

    inferred_kind = infer_manifest_kind(valid)
    chosen_mode = args.mode
    if chosen_mode == "auto":
        chosen_mode = "pairs" if inferred_kind == "pairs" else "sides"

    periods, period_to_idx = build_period_mapping(valid)
    print(f"\n📚 Loaded manifest: {len(valid)} valid records (total {manifest_length}).")
    print(f"🧭 Inferred manifest kind: {inferred_kind} | Running mode: {chosen_mode}")
    print(f"🏷️  Classes: {len(periods)}")

    if chosen_mode == "sides":
        # Ensure side records exist
        side_records = [r for r in valid if r.get("image_path")]
        train_records, val_records, train_counts = split_sides(side_records, seed=cfg.seed)
        class_weights = compute_class_weights(train_counts, periods, cfg.max_weight_cap)

        run = train_loop_sides(cfg, periods, period_to_idx, train_records, val_records, class_weights, args.output_prefix)

        export_embeddings_sides(
            cfg, records_all, periods, period_to_idx,
            run["model"], run["val_transform"], manifest_length, args.output_prefix
        )

    elif chosen_mode == "pairs":
        pair_records = [
            r for r in valid
            if r.get("period") and record_has_complete_pair(r)
        ]
        if not pair_records:
            raise RuntimeError("Mode 'pairs' requested, but no pair records found in manifest.")

        # Ensure coin_identity exists (fallback to manifest index)
        for r in pair_records:
            r["coin_identity"] = (
                r.get("coin_identity")
                or r.get("coin_id")
                or f"coin_{r['_manifest_index']}"
            )

        train_pairs, val_pairs, train_counts = split_pairs(pair_records, seed=cfg.seed)
        class_weights = compute_class_weights(train_counts, periods, cfg.max_weight_cap)

        run = train_loop_pairs(cfg, periods, period_to_idx, train_pairs, val_pairs, class_weights, args.output_prefix)

        export_embeddings_pairs(
            cfg, records_all, periods, period_to_idx,
            run["model"], run["val_transform"], manifest_length, args.output_prefix
        )

    else:
        raise ValueError(f"Unknown mode: {chosen_mode}")


if __name__ == "__main__":
    main()
