#!/usr/bin/env python3
"""
embed_full_dataset.py - Generate MobileNetV2 embeddings for ALL ml_dataset records.

Uses the trained trivalaya_model_v5.pth to embed every active image in the database,
producing embeddings compatible with cluster_coins.py --embeddings mode.

Usage:
    python embed_full_dataset.py
    python embed_full_dataset.py --model trivalaya_model_v5.pth --output embeddings_full
    python embed_full_dataset.py --gpu  # Use CUDA if available
"""

import os
import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
DB_HOST = os.getenv("TRIVALAYA_DB_HOST", "127.0.0.1")
DB_USER = os.getenv("TRIVALAYA_DB_USER", "auction_user")
DB_PASSWORD = os.getenv("TRIVALAYA_DB_PASSWORD", "Veritas@2024")
DB_NAME = os.getenv("TRIVALAYA_DB_NAME", "auction_data")

BATCH_SIZE = 64

def get_db_connection():
    """Create database connection."""
    import mysql.connector
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

def load_all_records():
    """Load all active records from ml_dataset."""
    print("📊 Loading all active records from database...")
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    query = """
        SELECT m.id, m.image_path, m.period, m.label_confidence, m.raw_label
        FROM ml_dataset m
        WHERE m.image_path IS NOT NULL 
        AND m.image_path != ''
        AND m.is_active = 1
        ORDER BY m.id
    """
    
    cursor.execute(query)
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    
    print(f"  Loaded {len(records)} active records")
    return records

def resolve_image_path(db_path):
    """Find actual file location."""
    path_obj = Path(db_path)
    if path_obj.is_absolute() and path_obj.exists():
        return path_obj

    candidates = [
        path_obj,
        Path(".") / db_path,
        Path("..") / db_path,
        Path("/root") / db_path,
        Path.home() / db_path,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def load_trained_model(model_path, meta_path, device):
    """Load the trained MobileNetV2 model."""
    print(f"🧠 Loading trained model from {model_path}...")
    
    # Load metadata to get number of classes
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Handle different metadata formats
    if 'idx_to_period' in meta:
        class_names = meta['idx_to_period']
        num_classes = len(class_names)
    elif 'period_to_idx' in meta:
        num_classes = len(meta['period_to_idx'])
        class_names = list(meta['period_to_idx'].keys())
    else:
        num_classes = meta.get('num_classes', len(meta.get('class_names', [])))
        class_names = meta.get('class_names', [])
    
    print(f"  Model has {num_classes} classes: {class_names}")
    
    # Build model architecture (must match training)
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes)
    )
    
    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"  ✅ Model loaded on {device}")
    return model, meta

def get_feature_extractor(model):
    """
    Create a feature extractor that returns the penultimate layer (1280-d).
    MobileNetV2 structure: features -> avgpool -> classifier
    We want the output after avgpool, before classifier.
    """
    class FeatureExtractor(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.features = base_model.features
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)  # (batch, 1280)
            return x
    
    return FeatureExtractor(model)

def get_transforms():
    """Get the same transforms used during training."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def embed_records(records, feature_extractor, transform, device):
    """
    Generate embeddings for all records.
    Returns embeddings array and list of valid records (those with loadable images).
    """
    print(f"\n🔍 Generating embeddings for {len(records)} records...")
    
    embeddings = []
    valid_records = []
    failed_paths = []
    
    batch_tensors = []
    batch_records = []
    
    for record in tqdm(records, desc="Processing images"):
        path = resolve_image_path(record['image_path'])
        if path is None:
            failed_paths.append(record['image_path'])
            continue
        
        try:
            img = Image.open(path).convert('RGB')
            tensor = transform(img)
            batch_tensors.append(tensor)
            batch_records.append(record)
            
            # Process batch
            if len(batch_tensors) >= BATCH_SIZE:
                batch_embeddings = process_batch(batch_tensors, feature_extractor, device)
                embeddings.extend(batch_embeddings)
                valid_records.extend(batch_records)
                batch_tensors = []
                batch_records = []
                
        except Exception as e:
            failed_paths.append(f"{record['image_path']} ({e})")
            continue
    
    # Process remaining batch
    if batch_tensors:
        batch_embeddings = process_batch(batch_tensors, feature_extractor, device)
        embeddings.extend(batch_embeddings)
        valid_records.extend(batch_records)
    
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # L2 normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    embeddings = embeddings / norms
    
    print(f"\n  ✅ Generated {len(embeddings)} embeddings")
    print(f"  ❌ Failed: {len(failed_paths)} images")
    
    if failed_paths and len(failed_paths) <= 20:
        print("  Failed paths:")
        for p in failed_paths[:20]:
            print(f"    - {p}")
    elif failed_paths:
        print(f"  (showing first 20 of {len(failed_paths)} failures)")
        for p in failed_paths[:20]:
            print(f"    - {p}")
    
    return embeddings, valid_records

def process_batch(tensors, feature_extractor, device):
    """Process a batch of image tensors through the feature extractor."""
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        features = feature_extractor(batch)
    return features.cpu().numpy().tolist()

def save_embeddings(embeddings, valid_records, output_prefix):
    """Save embeddings and metadata in cluster_coins.py compatible format."""
    print(f"\n💾 Saving embeddings...")
    
    # Save numpy array
    npy_path = f"{output_prefix}.npy"
    np.save(npy_path, embeddings)
    print(f"  ✅ {npy_path} ({embeddings.shape[0]} x {embeddings.shape[1]})")
    
    # Save metadata JSON
    meta = {
        "paths": [r['image_path'] for r in valid_records],
        "ids": [r['id'] for r in valid_records],
        "labels": [r['period'] or 'unknown' for r in valid_records],
        "count": len(valid_records),
        "embedding_dim": embeddings.shape[1],
    }
    
    meta_path = f"{output_prefix}_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  ✅ {meta_path}")
    
    # Print summary by period
    period_counts = Counter(r['period'] or 'unknown' for r in valid_records)
    print(f"\n📈 Coverage by period:")
    for period, count in sorted(period_counts.items(), key=lambda x: -x[1]):
        print(f"    {period}: {count}")
    
    return npy_path, meta_path

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for full ml_dataset")
    parser.add_argument('--model', type=str, default='trivalaya_model_v5.pth',
                        help="Path to trained model weights")
    parser.add_argument('--meta', type=str, default='trivalaya_model_v5_meta.json',
                        help="Path to model metadata JSON")
    parser.add_argument('--output', type=str, default='trivalaya_embeddings_full',
                        help="Output prefix for .npy and _meta.json files")
    parser.add_argument('--gpu', action='store_true', help="Use GPU if available")
    parser.add_argument('--limit', type=int, help="Limit to N records (for testing)")
    
    args = parser.parse_args()
    
    print("🪙 TRIVALAYA FULL DATASET EMBEDDING")
    print("=" * 50)
    
    # Setup device
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    # Load model
    model, model_meta = load_trained_model(args.model, args.meta, device)
    feature_extractor = get_feature_extractor(model)
    feature_extractor.eval()
    
    # Load all records
    records = load_all_records()
    
    if args.limit:
        print(f"⚠️  Limiting to {args.limit} records (test mode)")
        records = records[:args.limit]
    
    # Generate embeddings
    transform = get_transforms()
    embeddings, valid_records = embed_records(records, feature_extractor, transform, device)
    
    # Save results
    npy_path, meta_path = save_embeddings(embeddings, valid_records, args.output)
    
    print(f"\n✅ Done! Use with cluster_coins.py:")
    print(f"   python cluster_coins.py --embeddings {npy_path} --embeddings-meta {meta_path}")

if __name__ == "__main__":
    main()