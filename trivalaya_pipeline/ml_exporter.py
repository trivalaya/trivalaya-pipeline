"""
ML Dataset Exporter: Generate ML-ready datasets from processed coins.

Creates:
- Train/val/test splits with stratification
- Manifest files (JSON/CSV)
- Label mappings
- PyTorch Dataset class (Single & Paired)
"""

import json
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum

from .config import MLConfig, PathConfig, PERIOD_TAXONOMY
from .catalog import (
    CatalogDB, 
    MLDatasetEntry, 
    MLCoinDatasetEntry, 
    compute_image_hash, 
    compute_pair_hash
)
from .label_parser import LabelParser


class ExportMode(Enum):
    """Export mode selection."""
    DETECTION = "detection"      # One sample per detection (legacy/standard)
    COIN_PAIR = "coin_pair"      # One sample per coin (obverse + reverse)


class MissingSideRule(Enum):
    """How to handle coins with only one side detected (for COIN_PAIR mode)."""
    SKIP = "skip"           # Exclude from dataset (default - highest quality)
    DUPLICATE = "duplicate"  # Use available side for both
    PLACEHOLDER = "placeholder"  # Use neutral placeholder (requires placeholder image)


@dataclass
class ExportStats:
    """Statistics from dataset export (unified for both modes)."""
    
    total_images: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    duplicates_removed: int = 0
    duplicates_tracked: int = 0  # Tracked but inserted with flag
    needs_review: int = 0
    high_confidence: int = 0
    
    # Paired mode specific stats
    total_coins: int = 0
    coins_both_sides: int = 0
    coins_obv_only: int = 0
    coins_rev_only: int = 0
    coins_unknown_only: int = 0
    coins_skipped_missing_side: int = 0
    
    periods: Dict[str, int] = field(default_factory=dict)
    denominations: Dict[str, int] = field(default_factory=dict)
    materials: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)


class MLExporter:
    """Export processed coins to ML-ready dataset."""
    
    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        """Safely convert lot_number or similar fields to int."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    
    def __init__(
        self,
        db: CatalogDB,
        config: MLConfig = None,
        paths: PathConfig = None
    ):
        self.db = db
        self.config = config or MLConfig()
        self.paths = paths or PathConfig()
        self.parser = LabelParser()
        self._seen_hashes = set()
    
    def export_dataset(
        self,
        mode: ExportMode = ExportMode.DETECTION,
        min_likelihood: float = None,
        stratify_by: str = "period",
        missing_side_rule: MissingSideRule = MissingSideRule.SKIP,
        placeholder_path: str = None,
        seed: int = 42
    ) -> ExportStats:
        """
        Export full dataset with train/val/test splits.
        
        Args:
            mode: ExportMode.DETECTION or ExportMode.COIN_PAIR
            min_likelihood: Minimum coin_likelihood filter
            stratify_by: Field to stratify splits by
            missing_side_rule: (Pair mode only) Strategy for missing sides
            placeholder_path: (Pair mode only) Path to placeholder image
            seed: Random seed
        """
        # Dispatch to specific mode handler
        if mode == ExportMode.COIN_PAIR:
            return self.export_coin_pair_dataset(
                min_likelihood=min_likelihood,
                stratify_by=stratify_by,
                missing_side_rule=missing_side_rule,
                placeholder_path=placeholder_path,
                seed=seed
            )
        
        # Default: Detection Mode
        return self._export_detection_mode(
            min_likelihood=min_likelihood,
            stratify_by=stratify_by,
            seed=seed
        )

    # =========================================================================
    # DETECTION MODE (Legacy/Single Image)
    # =========================================================================

    def _export_detection_mode(
        self,
        min_likelihood: float = None,
        stratify_by: str = "period",
        seed: int = 42
    ) -> ExportStats:
        random.seed(seed)
        min_likelihood = min_likelihood or self.config.min_coin_likelihood
        stats = ExportStats()
        self._seen_hashes.clear()
        
        # Get exportable detections
        detections = self.db.get_exportable_detections(min_likelihood)
        
        if not detections:
            print("No detections to export")
            return stats
        
        print(f"Processing {len(detections)} detections...")
        
        entries = []
        for det in detections:
            label = self.parser.parse(
                det.get('title', ''),
                det.get('description', '')
            )
            
            normalized_path = det.get('normalized_path', '')
            if not normalized_path:
                continue
            
            # Deduplication only works with local files (needs pixel access)
            img_hash = ""
            
            entries.append({
                'detection_id': det['id'],
                'image_path': normalized_path,
                'image_hash': img_hash,
                'label': label,
                'site': det.get('site') or 'unknown',
                'auction_house': det.get('auction_house') or 'unknown',
                'sale_id': det.get('sale_id') or 'unknown',
                'lot_number': self._safe_int(det.get('lot_number')),
                'period': det.get('period'),  # From auction_info via lot range
            })
        
        if not entries:
            print("No valid entries after filtering")
            return stats
        
        # Stratified split
        train, val, test = self._stratified_split(
            entries, stratify_by,
            self.config.train_ratio, self.config.val_ratio, self.config.test_ratio
        )
        
        # Create output directory for manifests
        self.paths.ml_dataset.mkdir(parents=True, exist_ok=True)
        
        # Record each split (manifest-only, no file copying)
        for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
            self._record_split(split_data, split_name, stats)
        
        # Update counts
        stats.train_count = len(train)
        stats.val_count = len(val)
        stats.test_count = len(test)
        stats.total_images = len(train) + len(val) + len(test)
        
        # Write metadata
        self._write_manifests()
        self._write_summary(stats)
        self._write_label_mappings(stats)
        
        return stats

    def _stratified_split(self, entries, stratify_by, train_ratio, val_ratio, test_ratio):
        groups = defaultdict(list)
        for entry in entries:
            key = getattr(entry['label'], stratify_by, '') or 'unknown'
            groups[key].append(entry)
        
        train, val, test = [], [], []
        for group_entries in groups.values():
            random.shuffle(group_entries)
            n = len(group_entries)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train.extend(group_entries[:n_train])
            val.extend(group_entries[n_train:n_train + n_val])
            test.extend(group_entries[n_train + n_val:])
        
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)
        return train, val, test

    def _record_split(self, entries, split_name, stats):
        """Record entries into DB for manifest generation. No file copying."""
        for entry in entries:
            label = entry['label']
            period = entry.get('period') or label.period or 'unknown'
            
            # Stats
            if period != 'unknown':
                stats.periods[period] = stats.periods.get(period, 0) + 1
            if label.denomination:
                stats.denominations[label.denomination] = stats.denominations.get(label.denomination, 0) + 1
            if label.material:
                stats.materials[label.material] = stats.materials.get(label.material, 0) + 1
            
            if label.needs_review:
                stats.needs_review += 1
            if label.confidence > 0.7:
                stats.high_confidence += 1
            
            # Store Spaces key directly
            ml_entry = MLDatasetEntry(
                coin_detection_id=entry['detection_id'],
                image_path=entry['image_path'],
                image_hash=entry['image_hash'],
                split=split_name,
                period=period,
                subperiod=getattr(label, 'subperiod', None),
                authority=getattr(label, 'authority', None),
                denomination=label.denomination,
                mint=label.mint,
                material=label.material,
                raw_label=label.original_text,
                label_confidence=label.confidence,
                needs_review=label.needs_review,
            )
            self.db.insert_ml_entry(ml_entry)

    def _write_manifests(self):
        for split in ['train', 'val', 'test']:
            manifest = self.db.export_split_manifest(split)
            json_path = self.paths.ml_dataset / f'{split}_manifest.json'
            with open(json_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            csv_path = self.paths.ml_dataset / f'{split}_manifest.csv'
            if manifest:
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
                    writer.writeheader()
                    writer.writerows(manifest)

    def _write_summary(self, stats: ExportStats):
        summary = {
            'statistics': stats.to_dict(),
            'config': {
                'train_ratio': self.config.train_ratio,
                'val_ratio': self.config.val_ratio,
                'test_ratio': self.config.test_ratio,
                'min_coin_likelihood': self.config.min_coin_likelihood,
                'dedup_enabled': self.config.enable_dedup,
            },
            'taxonomy': {'periods': list(PERIOD_TAXONOMY.keys())}
        }
        summary_path = self.paths.ml_dataset / 'dataset_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def _write_label_mappings(self, stats: ExportStats):
        mappings = {
            'period': {p: i for i, p in enumerate(sorted(stats.periods.keys()))},
            'denomination': {d: i for i, d in enumerate(sorted(stats.denominations.keys()))},
            'material': {m: i for i, m in enumerate(sorted(stats.materials.keys()))},
        }
        mapping_path = self.paths.ml_dataset / 'label_mappings.json'
        with open(mapping_path, 'w') as f:
            json.dump(mappings, f, indent=2)

    # =========================================================================
    # COIN PAIR MODE (Obverse + Reverse)
    # =========================================================================

    def export_coin_pair_dataset(
        self,
        min_likelihood: float = None,
        stratify_by: str = "period",
        missing_side_rule: MissingSideRule = MissingSideRule.SKIP,
        placeholder_path: str = None,
        seed: int = 42
    ) -> ExportStats:
        random.seed(seed)
        min_likelihood = min_likelihood or self.config.min_coin_likelihood
        stats = ExportStats()
        self.missing_side_rule = missing_side_rule
        
        if missing_side_rule == MissingSideRule.PLACEHOLDER and not placeholder_path:
            raise ValueError("placeholder_path required when missing_side_rule is PLACEHOLDER")
        
        # Get exportable coins
        coins = self.db.get_exportable_coins(min_likelihood)
        
        if not coins:
            print("No coins to export")
            return stats
        
        print(f"Processing {len(coins)} coins in PAIR mode...")
        
        entries = []
        
        for coin in coins:
            obv_path = coin.get('obv_path')
            rev_path = coin.get('rev_path')
            unk_path = coin.get('unk_path')
            
            has_obv = bool(obv_path)
            has_rev = bool(rev_path)
            has_unk = bool(unk_path)
            
            # Resolve Paths based on Rule
            final_obv, final_rev = None, None
            
            if has_obv and has_rev:
                final_obv, final_rev = obv_path, rev_path
                stats.coins_both_sides += 1
            elif has_obv and not has_rev:
                stats.coins_obv_only += 1
                if missing_side_rule == MissingSideRule.SKIP:
                    stats.coins_skipped_missing_side += 1
                    continue
                elif missing_side_rule == MissingSideRule.DUPLICATE:
                    final_obv, final_rev = obv_path, obv_path
                else: # PLACEHOLDER
                    final_obv, final_rev = obv_path, placeholder_path
            elif has_rev and not has_obv:
                stats.coins_rev_only += 1
                if missing_side_rule == MissingSideRule.SKIP:
                    stats.coins_skipped_missing_side += 1
                    continue
                elif missing_side_rule == MissingSideRule.DUPLICATE:
                    final_obv, final_rev = rev_path, rev_path
                else: # PLACEHOLDER
                    final_obv, final_rev = placeholder_path, rev_path
            elif has_unk:
                stats.coins_unknown_only += 1
                if missing_side_rule == MissingSideRule.SKIP:
                    stats.coins_skipped_missing_side += 1
                    continue
                elif missing_side_rule == MissingSideRule.DUPLICATE:
                    final_obv, final_rev = unk_path, unk_path
                else: # PLACEHOLDER - Treat unk as obv
                    final_obv, final_rev = unk_path, placeholder_path
            else:
                continue

            # Hash-based dedup disabled for Spaces paths (needs pixel access)
            obv_hash = ""
            rev_hash = ""
            pair_hash = ""
            
            is_duplicate = False
            duplicate_of_coin_id = None
            
            label = self.parser.parse(coin.get('title', ''), coin.get('description', ''))
            
            entries.append({
                'coin_id': coin['coin_id'],
                'obv_path': final_obv,
                'rev_path': final_rev,
                'has_obv': has_obv,
                'has_rev': has_rev,
                'obv_hash': obv_hash,
                'rev_hash': rev_hash,
                'pair_hash': pair_hash,
                'is_duplicate': is_duplicate,
                'duplicate_of_coin_id': duplicate_of_coin_id,
                'label': label,
                'auction_house': coin.get('auction_house') or 'unknown',
                'sale_id': coin.get('sale_id') or 'unknown',
                'lot_number': self._safe_int(coin.get('lot_number')),
                'period': coin.get('auction_period') or label.period or 'unknown',
            })
            
        # Split logic (only non-duplicates)
        non_dup_entries = [e for e in entries if not e['is_duplicate']]
        dup_entries = [e for e in entries if e['is_duplicate']]
        
        if not non_dup_entries:
            print("No valid non-duplicate entries after filtering")
            return stats
        
        stats.total_coins = len(non_dup_entries)
        
        train, val, test = self._stratified_split_coins(
            non_dup_entries, stratify_by,
            self.config.train_ratio, self.config.val_ratio, self.config.test_ratio
        )
        
        self.paths.ml_dataset.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
            self._record_coin_pair_split(split_data, split_name, stats)
            
        # Insert Duplicates (Flagged) — assign to same split as their primary
        split_map = {}
        for e in train: split_map[e['coin_id']] = 'train'
        for e in val:   split_map[e['coin_id']] = 'val'
        for e in test:  split_map[e['coin_id']] = 'test'
        
        for entry in dup_entries:
            primary_id = entry['duplicate_of_coin_id']
            dup_split = split_map.get(primary_id, 'train')
            self._record_single_coin_pair(entry, dup_split, stats, is_duplicate=True)
            
        stats.train_count = len(train)
        stats.val_count = len(val)
        stats.test_count = len(test)
        stats.total_images = (len(train) + len(val) + len(test)) * 2
        
        self._write_coin_pair_manifests()
        self._write_coin_pair_summary(stats, missing_side_rule)
        self._write_label_mappings(stats)
        
        return stats

    def _stratified_split_coins(self, entries, stratify_by, train_ratio, val_ratio, test_ratio):
        groups = defaultdict(list)
        for entry in entries:
            key = entry.get('period', 'unknown') if stratify_by == 'period' else \
                  getattr(entry['label'], stratify_by, 'unknown') or 'unknown'
            groups[key].append(entry)
            
        train, val, test = [], [], []
        for group_entries in groups.values():
            random.shuffle(group_entries)
            n = len(group_entries)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train.extend(group_entries[:n_train])
            val.extend(group_entries[n_train:n_train + n_val])
            test.extend(group_entries[n_train + n_val:])
        
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)
        return train, val, test

    def _record_coin_pair_split(self, entries, split_name, stats):
        """Record a list of coin pair entries into DB. No file copying."""
        for entry in entries:
            self._record_single_coin_pair(entry, split_name, stats, is_duplicate=False)

    def _record_single_coin_pair(self, entry, split_name, stats, is_duplicate=False):
        """Record one coin pair into DB with Spaces keys. No file copying."""
        obv_path = entry.get('obv_path')
        rev_path = entry.get('rev_path')
        
        if not obv_path or not rev_path:
            print(f"  ⚠ Skipping coin {entry.get('coin_id')}: incomplete pair reached export")
            return
        
        label = entry['label']
        resolved_period = entry['period']
        
        # Stats update (only for non-duplicates)
        if not is_duplicate:
            if resolved_period and resolved_period != 'unknown':
                stats.periods[resolved_period] = stats.periods.get(resolved_period, 0) + 1
            if label.denomination:
                stats.denominations[label.denomination] = stats.denominations.get(label.denomination, 0) + 1
            if label.material:
                stats.materials[label.material] = stats.materials.get(label.material, 0) + 1
            
            if label.needs_review: stats.needs_review += 1
            if label.confidence > 0.7: stats.high_confidence += 1
        
        # Store Spaces keys directly
        ml_entry = MLCoinDatasetEntry(
            coin_id=entry['coin_id'],
            obv_path=obv_path,
            rev_path=rev_path,
            has_obv=entry['has_obv'],
            has_rev=entry['has_rev'],
            obv_hash=entry['obv_hash'],
            rev_hash=entry['rev_hash'],
            pair_hash=entry['pair_hash'],
            is_duplicate=is_duplicate,
            duplicate_of_coin_id=entry.get('duplicate_of_coin_id'),
            split=split_name,
            period=resolved_period,
            subperiod=getattr(label, 'subperiod', None),
            authority=getattr(label, 'authority', None),
            denomination=label.denomination,
            mint=label.mint,
            material=label.material,
            raw_label=label.original_text,
            label_confidence=label.confidence,
            needs_review=label.needs_review,
        )
        self.db.insert_ml_coin_entry(ml_entry)

    def _write_coin_pair_manifests(self):
        for split in ['train', 'val', 'test']:
            manifest = self.db.export_coin_pair_manifest(split)
            json_path = self.paths.ml_dataset / f'{split}_pairs_manifest.json'
            with open(json_path, 'w') as f: json.dump(manifest, f, indent=2)
            
            csv_path = self.paths.ml_dataset / f'{split}_pairs_manifest.csv'
            if manifest:
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
                    writer.writeheader()
                    writer.writerows(manifest)

    def _write_coin_pair_summary(self, stats: ExportStats, missing_side_rule: MissingSideRule):
        summary = {
            'mode': 'coin_pair',
            'missing_side_rule': missing_side_rule.value,
            'statistics': stats.to_dict(),
            'config': {
                'train_ratio': self.config.train_ratio,
                'val_ratio': self.config.val_ratio,
                'test_ratio': self.config.test_ratio,
                'min_coin_likelihood': self.config.min_coin_likelihood,
                'dedup_enabled': self.config.enable_dedup,
            },
            'pairing': {
                'both_sides': stats.coins_both_sides,
                'obv_only': stats.coins_obv_only,
                'rev_only': stats.coins_rev_only,
                'unknown_only': stats.coins_unknown_only,
                'skipped_missing_side': stats.coins_skipped_missing_side,
            },
            'taxonomy': {'periods': list(PERIOD_TAXONOMY.keys())}
        }
        summary_path = self.paths.ml_dataset / 'dataset_pairs_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def generate_pytorch_dataset(self) -> str:
        """Generate PyTorch Dataset classes (Unified: Single & Paired)."""
        code = '''"""
Auto-generated PyTorch Dataset for Trivalaya coin dataset.
Includes single-image and paired (obverse/reverse) variants with synced transforms.
"""

import json
import random
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image


class SyncedTransform:
    """
    Wrapper that applies geometric transforms consistently to paired images.
    
    Geometric transforms (flip, rotation, crop) are synced between obverse and reverse.
    Photometric transforms (color, contrast) can be applied independently.
    """
    
    def __init__(
        self,
        geometric_transforms: list = None,
        photometric_transforms: list = None,
        final_transforms: list = None,
    ):
        self.geometric_transforms = geometric_transforms or []
        self.photometric_transforms = photometric_transforms or []
        self.final_transforms = final_transforms or []
    
    def __call__(self, obv_img: Image.Image, rev_img: Image.Image) -> Tuple[Any, Any]:
        # Apply synced geometric transforms
        for t in self.geometric_transforms:
            obv_img, rev_img = self._apply_synced(t, obv_img, rev_img)
        
        # Apply independent photometric transforms
        for t in self.photometric_transforms:
            obv_img = t(obv_img)
            rev_img = t(rev_img)
        
        # Apply final transforms (ToTensor, Normalize)
        for t in self.final_transforms:
            obv_img = t(obv_img)
            rev_img = t(rev_img)
        
        return obv_img, rev_img
    
    def _apply_synced(self, transform, obv_img, rev_img):
        """Apply a transform with same random state to both images."""
        state = random.getstate()
        torch_state = torch.get_rng_state()
        
        obv_img = transform(obv_img)
        
        # Restore and apply same randomness
        random.setstate(state)
        torch.set_rng_state(torch_state)
        
        rev_img = transform(rev_img)
        return obv_img, rev_img


class CoinDataset(Dataset):
    """PyTorch Dataset for single-image coin classification."""
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_field: str = 'period',
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_field = target_field
        
        with open(self.root / f'{split}_manifest.json') as f:
            self.samples = json.load(f)
        
        with open(self.root / 'label_mappings.json') as f:
            self.label_mappings = json.load(f)
        
        self.class_to_idx = self.label_mappings.get(target_field, {})
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        image = Image.open(self.root / sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        label_str = sample.get(self.target_field, '')
        label = self.class_to_idx.get(label_str, -1)
        return image, label


class CoinPairDataset(Dataset):
    """
    PyTorch Dataset for paired (obverse/reverse) coin classification.
    Returns (obv_image, rev_image, label) tuples.
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        synced_transform: Optional[SyncedTransform] = None,
        target_field: str = 'period',
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.synced_transform = synced_transform
        self.target_field = target_field
        
        with open(self.root / f'{split}_pairs_manifest.json') as f:
            self.samples = json.load(f)
        
        with open(self.root / 'label_mappings.json') as f:
            self.label_mappings = json.load(f)
        
        self.class_to_idx = self.label_mappings.get(target_field, {})
        self.num_classes = len(self.class_to_idx)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]
        obv_image = Image.open(self.root / sample['obv_path']).convert('RGB')
        rev_image = Image.open(self.root / sample['rev_path']).convert('RGB')
        
        if self.synced_transform:
            obv_image, rev_image = self.synced_transform(obv_image, rev_image)
        elif self.transform:
            obv_image = self.transform(obv_image)
            rev_image = self.transform(rev_image)
        
        label_str = sample.get(self.target_field, '')
        label = self.class_to_idx.get(label_str, -1)
        return obv_image, rev_image, label
'''
        
        output_path = self.paths.ml_dataset / 'coin_dataset.py'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(code)
        
        return str(output_path)