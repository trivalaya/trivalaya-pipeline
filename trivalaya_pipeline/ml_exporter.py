"""
ML Dataset Exporter: Generate ML-ready datasets from processed coins.

Creates:
- Train/val/test splits with stratification
- Manifest files (JSON/CSV)
- Label mappings
- PyTorch Dataset class
"""

import json
import csv
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict

from .config import MLConfig, PathConfig, PERIOD_TAXONOMY
from .catalog import CatalogDB, MLDatasetEntry, compute_image_hash
from .label_parser import LabelParser


@dataclass
class ExportStats:
    """Statistics from dataset export."""
    
    total_images: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    duplicates_removed: int = 0
    needs_review: int = 0
    high_confidence: int = 0
    
    periods: Dict[str, int] = None
    denominations: Dict[str, int] = None
    materials: Dict[str, int] = None
    
    def __post_init__(self):
        self.periods = self.periods or {}
        self.denominations = self.denominations or {}
        self.materials = self.materials or {}
    
    def to_dict(self) -> dict:
        return asdict(self)


class MLExporter:
    """Export processed coins to ML-ready dataset."""
    
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
        min_likelihood: float = None,
        stratify_by: str = "period",
        seed: int = 42
    ) -> ExportStats:
        """
        Export full dataset with train/val/test splits.
        """
        random.seed(seed)
        min_likelihood = min_likelihood or self.config.min_coin_likelihood
        stats = ExportStats()
        
        # Get exportable detections
        detections = self.db.get_exportable_detections(min_likelihood)
        
        if not detections:
            print("No detections to export")
            return stats
        
        print(f"Processing {len(detections)} detections...")
        
        # Parse labels and prepare entries
        entries = []
        for det in detections:
            label = self.parser.parse(
                det.get('title', ''),
                det.get('description', '')
            )
            
            # Check image exists
            normalized_path = det.get('normalized_path', '')
            if not normalized_path or not Path(normalized_path).exists():
                continue
            
            # Deduplication
            if self.config.enable_dedup:
                img_hash = compute_image_hash(normalized_path, self.config.hash_algorithm)
                if img_hash in self._seen_hashes:
                    stats.duplicates_removed += 1
                    continue
                self._seen_hashes.add(img_hash)
            else:
                img_hash = ""
            
            entries.append({
                'detection_id': det['id'],
                'image_path': normalized_path,
                'image_hash': img_hash,
                'label': label,
                'site': det.get('site', ''),
                'auction_id': det.get('auction_id', ''),
                'lot_number': det.get('lot_number', 0),
            })
        
        if not entries:
            print("No valid entries after filtering")
            return stats
        
        # Stratified split
        train, val, test = self._stratified_split(
            entries, stratify_by,
            self.config.train_ratio,
            self.config.val_ratio,
            self.config.test_ratio
        )
        
        # Create output directories
        self.paths.create_directories()
        
        # Export each split
        for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
            split_dir = getattr(self.paths, f'{split_name}_split')
            self._export_split(split_data, split_dir, split_name, stats)
        
        # Update counts
        stats.train_count = len(train)
        stats.val_count = len(val)
        stats.test_count = len(test)
        stats.total_images = len(train) + len(val) + len(test)
        
        # Write manifests and metadata
        self._write_manifests()
        self._write_summary(stats)
        self._write_label_mappings(stats)
        
        return stats
    
    def _stratified_split(
        self,
        entries: List[Dict],
        stratify_by: str,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float
    ) -> Tuple[List, List, List]:
        """Split data with stratification."""
        groups = defaultdict(list)
        for entry in entries:
            # Safely get attribute, default to 'unknown'
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
    
    def _export_split(
        self,
        entries: List[Dict],
        output_dir: Path,
        split_name: str,
        stats: ExportStats
    ):
        """Export a single split."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for entry in entries:
            src_path = Path(entry['image_path'])
            if not src_path.exists():
                continue
            
            label = entry['label']
            
            # Organize by period
            period_dir = output_dir / (label.period or 'unknown')
            period_dir.mkdir(exist_ok=True)
            
            # Copy image
            dst_name = f"{entry['site']}_{entry['auction_id']}_{entry['lot_number']:05d}.jpg"
            dst_path = period_dir / dst_name
            shutil.copy2(src_path, dst_path)
            
            # Update stats
            if label.period:
                stats.periods[label.period] = stats.periods.get(label.period, 0) + 1
            if label.denomination:
                stats.denominations[label.denomination] = stats.denominations.get(label.denomination, 0) + 1
            if label.material:
                stats.materials[label.material] = stats.materials.get(label.material, 0) + 1
            
            # FIXED: Updated attribute names to match LabelParser
            if label.needs_review:
                stats.needs_review += 1
            
            # FIXED: Use 'confidence' instead of 'overall_confidence'
            if label.confidence > 0.7:
                stats.high_confidence += 1
            
            # Save to database
            ml_entry = MLDatasetEntry(
                coin_detection_id=entry['detection_id'],
                image_path=str(dst_path),
                image_hash=entry['image_hash'],
                split=split_name,
                period=label.period,
                # FIXED: Use getattr for optional fields not yet in parser
                subperiod=getattr(label, 'subperiod', None),
                authority=getattr(label, 'authority', None),
                denomination=label.denomination,
                mint=label.mint,
                material=label.material,
                # FIXED: Map original_text to raw_label
                raw_label=label.original_text, 
                label_confidence=label.confidence,
                needs_review=label.needs_review,
            )
            self.db.insert_ml_entry(ml_entry)
    
    def _write_manifests(self):
        """Write manifest files for each split."""
        for split in ['train', 'val', 'test']:
            manifest = self.db.export_split_manifest(split)
            
            # JSON
            json_path = self.paths.ml_dataset / f'{split}_manifest.json'
            with open(json_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # CSV
            csv_path = self.paths.ml_dataset / f'{split}_manifest.csv'
            if manifest:
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
                    writer.writeheader()
                    writer.writerows(manifest)
    
    def _write_summary(self, stats: ExportStats):
        """Write dataset summary."""
        summary = {
            'statistics': stats.to_dict(),
            'config': {
                'train_ratio': self.config.train_ratio,
                'val_ratio': self.config.val_ratio,
                'test_ratio': self.config.test_ratio,
                'min_coin_likelihood': self.config.min_coin_likelihood,
                'dedup_enabled': self.config.enable_dedup,
            },
            'taxonomy': {
                'periods': list(PERIOD_TAXONOMY.keys()),
            }
        }
        
        summary_path = self.paths.ml_dataset / 'dataset_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _write_label_mappings(self, stats: ExportStats):
        """Write label-to-index mappings for training."""
        mappings = {
            'period': {p: i for i, p in enumerate(sorted(stats.periods.keys()))},
            'denomination': {d: i for i, d in enumerate(sorted(stats.denominations.keys()))},
            'material': {m: i for i, m in enumerate(sorted(stats.materials.keys()))},
        }
        
        mapping_path = self.paths.ml_dataset / 'label_mappings.json'
        with open(mapping_path, 'w') as f:
            json.dump(mappings, f, indent=2)
    
    def generate_pytorch_dataset(self) -> str:
        """Generate a PyTorch Dataset class file."""
        code = '''"""
Auto-generated PyTorch Dataset for Trivalaya coin dataset.
"""

import json
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image


class CoinDataset(Dataset):
    """PyTorch Dataset for coin classification."""
    
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
        
        # Load manifest
        with open(self.root / f'{split}_manifest.json') as f:
            self.samples = json.load(f)
        
        # Load label mappings
        with open(self.root / 'label_mappings.json') as f:
            self.label_mappings = json.load(f)
        
        self.class_to_idx = self.label_mappings.get(target_field, {})
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        label_str = sample.get(self.target_field, '')
        label = self.class_to_idx.get(label_str, -1)
        
        return image, label


class MultiLabelCoinDataset(Dataset):
    """Multi-label variant returning all classification targets."""
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        
        with open(self.root / f'{split}_manifest.json') as f:
            self.samples = json.load(f)
        
        with open(self.root / 'label_mappings.json') as f:
            self.label_mappings = json.load(f)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, int]]:
        sample = self.samples[idx]
        
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        labels = {}
        for field in ['period', 'denomination', 'material']:
            mapping = self.label_mappings.get(field, {})
            labels[field] = mapping.get(sample.get(field, ''), -1)
        
        return image, labels
'''
        
        output_path = self.paths.ml_dataset / 'coin_dataset.py'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(code)
        
        return str(output_path)