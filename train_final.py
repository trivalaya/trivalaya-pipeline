"""
ML POC v5.2 (Gold Master): Fully deterministic, production logging, 
optimized throughput, and manifest-aligned embedding export.

Changes from v5.1:
- Moved CUBLAS_WORKSPACE_CONFIG before torch import (critical for determinism)
- Fixed O(N²) path lookup → O(1) dict lookup in embedding export
- Removed weights_only=True for torch version compatibility
- Conditional prefetch_factor (only included when num_workers > 0)
- PIL context manager to prevent file handle buildup
"""
# --- ENVIRONMENT SETUP (must be before torch import) ---
import os

SEED = 42
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(SEED)

# --- IMPORTS ---
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image

# --- DETERMINISM SETUP (after torch import) ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

# This will throw if any op is nondeterministic - exactly what we want for gold master
torch.use_deterministic_algorithms(True)

# --- CONFIG ---
CONFIG = {
    "batch_size": 16,
    "epochs": 20,
    "lr": 0.0005,
    "max_weight_cap": 10.0,
    "image_size": 128,
    "aug_rotation": 15,
    "aug_scale": (0.9, 1.1),
    "num_workers": 2,
    "pin_memory": True if torch.cuda.is_available() else False,
    "persistent_workers": True,
    "prefetch_factor": 2,
}

DATA_DIR = Path("trivalaya_data/03_ml_ready/dataset")
MANIFEST_FILE = DATA_DIR / "train_manifest.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output paths
MODEL_PATH = "trivalaya_model_v5.pth"
META_PATH = "trivalaya_model_v5_meta.json"
SKIP_LOG_PATH = "trivalaya_skipped_images.txt"
EMBEDDINGS_PATH = "trivalaya_embeddings.npy"
EMBEDDINGS_META_PATH = "trivalaya_embeddings_meta.json"

# --- LOGGING ---
skipped_files = set()


def load_data_from_bridge():
    """Load manifest and stamp each record with its manifest index."""
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"❌ Manifest not found: {MANIFEST_FILE}")
    
    with open(MANIFEST_FILE) as f:
        raw_data = json.load(f)
    
    # Stamp each record with its original manifest index
    for i, x in enumerate(raw_data):
        x["_manifest_index"] = i
    
    # Filter valid data (must have period)
    valid_data = [x for x in raw_data if x.get("period")]
    print(f"📚 Loaded {len(valid_data)} valid records from manifest (total: {len(raw_data)}).")
    return valid_data, len(raw_data)


def load_trivalaya_artifacts(embeddings_path, meta_path):
    """
    Load embeddings and metadata for downstream use.
    Returns (vectors, paths, labels, manifest_indices).
    
    manifest_indices allows mapping embeddings back to original manifest rows
    even when some images were skipped.
    """
    vectors = np.load(embeddings_path)
    with open(meta_path) as f:
        meta = json.load(f)
    
    paths = meta["paths"]
    labels = meta["labels"]
    manifest_indices = meta.get("manifest_indices", None)
    
    if len(paths) != vectors.shape[0]:
        raise ValueError(f"Mismatch: {len(paths)} paths vs {vectors.shape[0]} vectors")
    if len(labels) != vectors.shape[0]:
        raise ValueError(f"Mismatch: {len(labels)} labels vs {vectors.shape[0]} vectors")
    if manifest_indices is not None and len(manifest_indices) != vectors.shape[0]:
        raise ValueError(f"Mismatch: {len(manifest_indices)} indices vs {vectors.shape[0]} vectors")
    
    return vectors, paths, labels, manifest_indices


# 1. Load Data
try:
    all_data, manifest_length = load_data_from_bridge()
except FileNotFoundError as e:
    print(e)
    exit(1)

# 2. Split Data (Inclusive Strategy)
by_period = defaultdict(list)
for x in all_data:
    by_period[x["period"]].append(x)

train_data, val_data = [], []
train_counts = defaultdict(int)

# Sort for deterministic class mapping
sorted_periods = sorted(by_period.keys())
periods = [] 

print("\n📊 Data Distribution:")
for period in sorted_periods:
    items = by_period[period]
    count = len(items)
    if count == 0:
        continue
        
    # Shuffle specific to this block, deterministic via random.seed(42)
    random.shuffle(items)
    periods.append(period)

    # Adaptive Split
    if count == 1:
        train_data.extend(items)
        train_counts[period] += 1
    else:
        n_val = max(1, int(count * 0.2))
        val_data.extend(items[:n_val])
        train_data.extend(items[n_val:])
        train_counts[period] += (count - n_val)

period_to_idx = {p: i for i, p in enumerate(periods)}
print(f"✅ Final: {len(train_data)} Training samples | {len(val_data)} Validation samples")
print(f"   Classes: {len(periods)}")

# 3. Calculate Capped Class Weights
class_weights = []
total_train = len(train_data)

for p in periods:
    cnt = train_counts[p]
    if cnt == 0:
        weight = 0.0
    else:
        raw_weight = total_train / (len(periods) * cnt)
        weight = min(raw_weight, CONFIG["max_weight_cap"])
    class_weights.append(weight)

class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
print(f"⚖️  Weights calculated (Min: {min(class_weights):.2f}, Max: {max(class_weights):.2f})")


# 4. Robust Dataset Class
class CoinDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        path = item["image_path"]
        
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")

            # Use context manager to prevent file handle buildup
            with Image.open(path) as im:
                img = im.convert("RGB")
            
            label = period_to_idx[item["period"]]
            manifest_idx = item["_manifest_index"]
            return self.transform(img), label, manifest_idx
            
        except Exception as e:
            skipped_files.add(path)
            return None


def safe_collate(batch):
    """Filter out None values from failed loads."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.Tensor(), torch.Tensor(), torch.Tensor()
    return torch.utils.data.dataloader.default_collate(batch)


# Determinism for DataLoader workers
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

# 5. Coin-Safe Augmentation
train_transform = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.RandomAffine(
        degrees=CONFIG["aug_rotation"], 
        translate=(0.05, 0.05), 
        scale=CONFIG["aug_scale"]
    ),
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# DataLoader kwargs (conditional prefetch_factor)
loader_kwargs = {
    "collate_fn": safe_collate,
    "num_workers": CONFIG["num_workers"],
    "pin_memory": CONFIG["pin_memory"],
    "worker_init_fn": seed_worker,
    "generator": g,
}

if CONFIG["num_workers"] > 0:
    loader_kwargs["persistent_workers"] = CONFIG["persistent_workers"]
    loader_kwargs["prefetch_factor"] = CONFIG["prefetch_factor"]

train_loader = DataLoader(
    CoinDataset(train_data, train_transform), 
    batch_size=CONFIG["batch_size"], 
    shuffle=True,
    **loader_kwargs
)

val_loader = DataLoader(
    CoinDataset(val_data, val_transform), 
    batch_size=CONFIG["batch_size"], 
    shuffle=False,
    **loader_kwargs
)

# 6. Model Setup
print(f"🚀 Training on {DEVICE}...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, len(periods))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

# 7. Training Loop
best_val = 0.0

for epoch in range(CONFIG["epochs"]):
    model.train()
    train_loss = 0
    train_correct = 0
    train_steps = 0
    total_train_samples = 0
    
    for batch in train_loader:
        imgs, labels, _ = batch
        
        if imgs.numel() == 0 or labels.numel() == 0:
            continue
            
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()
        total_train_samples += labels.size(0)
        train_steps += 1
    
    # Validation
    model.eval()
    val_correct = 0
    total_val_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            imgs, labels, _ = batch
            if imgs.numel() == 0:
                continue
            
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            val_correct += (model(imgs).argmax(1) == labels).sum().item()
            total_val_samples += labels.size(0)
    
    val_acc = 0
    if total_val_samples > 0:
        val_acc = val_correct / total_val_samples * 100
        
    avg_loss = train_loss / max(1, train_steps)
    train_acc = train_correct / max(1, total_train_samples) * 100
    
    print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.3f} | Train: {train_acc:.0f}% | Val: {val_acc:.0f}%")
    
    if val_acc >= best_val:
        best_val = val_acc
        torch.save(model.state_dict(), MODEL_PATH)

# 8. Save Metadata & Logs
if skipped_files:
    with open(SKIP_LOG_PATH, "w") as f:
        for p in sorted(skipped_files):
            f.write(f"{p}\n")
    print(f"⚠️  WARNING: {len(skipped_files)} images failed. Log: {SKIP_LOG_PATH}")

meta = {
    "config": CONFIG,
    "period_to_idx": period_to_idx,
    "idx_to_period": periods,
    "best_val_acc": best_val,
    "skipped_count": len(skipped_files),
    "skipped_log": SKIP_LOG_PATH
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n💾 Model saved to {MODEL_PATH}")
print(f"💾 Metadata saved to {META_PATH}")
# ... [After Step 8: Save Metadata & Logs] ...

# 8.5. Detailed Evaluation (Confusion Matrix)
print("\n🔍 Generating Confusion Matrix on Validation Set...")

# Ensure we have the best model loaded
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for batch in val_loader:
        imgs, labels, _ = batch
        if imgs.numel() == 0:
            continue
            
        imgs = imgs.to(DEVICE, non_blocking=True)
        # Get predictions
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_targets.extend(labels.numpy())

# Calculate metrics
try:
    from sklearn.metrics import confusion_matrix, classification_report
    
    # 1. Print Console Report
    print("\n" + "="*30)
    print("CLASSIFICATION REPORT")
    print("="*30)
    report = classification_report(all_targets, all_preds, target_names=periods)
    print(report)

    # 2. Save Matrix to File (Raw Text)
    cm = confusion_matrix(all_targets, all_preds)
    matrix_save_path = "trivalaya_confusion_matrix.txt"
    
    with open(matrix_save_path, "w") as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix (Rows=True, Cols=Pred):\n")
        # Helper to format the matrix with labels
        f.write(f"{'':<20} " + " ".join([f"{p[:4]:>6}" for p in periods]) + "\n")
        for i, row in enumerate(cm):
            f.write(f"{periods[i]:<20} " + " ".join([f"{x:>6}" for x in row]) + "\n")
            
    print(f"✅ Confusion matrix saved to {matrix_save_path}")

except ImportError:
    print("⚠️  sklearn not found. Skipping detailed confusion matrix.")
    # Fallback: Simple Accuracy Per Class
    correct = np.array(all_preds) == np.array(all_targets)
    print(f"Global Validation Accuracy: {correct.mean():.2%}")

# ... [Proceed to Step 9: Generate Embeddings] ...

# 9. Generate GLOBAL Embeddings (Manifest-Aligned)
print("\n🧬 Generating Global Embeddings (L2 Normalized, Manifest-Aligned)...")

# Load best model for inference (no weights_only for compatibility)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

feature_extractor = torch.nn.Sequential(
    model.features,
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(1),
).to(DEVICE).eval()

# Use ALL valid data (preserving manifest order)
full_items = [x for x in all_data if x.get("period")]
full_dataset = CoinDataset(full_items, val_transform)

# Build O(1) lookup for manifest index → path
manifest_to_path = {x["_manifest_index"]: x["image_path"] for x in full_items}

# Create loader without persistent_workers for single pass
full_loader_kwargs = {
    "collate_fn": safe_collate,
    "num_workers": CONFIG["num_workers"],
    "pin_memory": CONFIG["pin_memory"],
    "worker_init_fn": seed_worker,
}

full_loader = DataLoader(
    full_dataset, 
    batch_size=CONFIG["batch_size"], 
    shuffle=False,  # Must be False to keep index alignment
    **full_loader_kwargs
)

embeddings = []
labels_list = []
paths_list = []
manifest_indices = []

with torch.no_grad():
    for imgs, lbls, indices in full_loader:
        if imgs.numel() == 0:
            continue
        
        imgs = imgs.to(DEVICE, non_blocking=True)
        feats = feature_extractor(imgs)  # (B, 1280)
        
        # L2 Normalize
        feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-12)
        
        embeddings.append(feats.cpu().numpy())
        labels_list.extend([periods[l] for l in lbls.tolist()])
        
        # O(1) path lookup via dict
        idx_list = indices.tolist()
        paths_list.extend([manifest_to_path[i] for i in idx_list])
        manifest_indices.extend(idx_list)

if embeddings:
    all_embeddings = np.concatenate(embeddings, axis=0)
    np.save(EMBEDDINGS_PATH, all_embeddings)
    
    # Save Extended Metadata with manifest alignment info
    with open(EMBEDDINGS_META_PATH, "w") as f:
        json.dump({
            "paths": paths_list,
            "labels": labels_list,
            "manifest_indices": manifest_indices,
            "manifest_length": manifest_length,
            "period_to_idx": period_to_idx,
            "idx_to_period": periods,
            "model_path": MODEL_PATH,
            "meta_path": META_PATH,
            "config": CONFIG,
            "backbone": "mobilenet_v2",
            "feature_dim": 1280,
            "normalized": True,
            "count": all_embeddings.shape[0]
        }, f, indent=2)
        
    print(f"✅ Saved {all_embeddings.shape[0]} embeddings to {EMBEDDINGS_PATH}")
    print(f"   Manifest coverage: {all_embeddings.shape[0]}/{manifest_length} rows")
    print(f"   Aligned metadata in {EMBEDDINGS_META_PATH}")
