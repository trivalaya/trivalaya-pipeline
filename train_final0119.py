"""ML POC with proper 80/20 split using Bridge Manifest"""
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
from collections import defaultdict
import os

# --- CONFIG ---
random.seed(42)
BATCH_SIZE = 16
EPOCHS = 20  # Increased slightly for better convergence
DATA_DIR = Path("trivalaya_data/03_ml_ready/dataset")
MANIFEST_FILE = DATA_DIR / "train_manifest.json"

def load_data_from_bridge():
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"❌ Manifest not found: {MANIFEST_FILE}")
    
    with open(MANIFEST_FILE) as f:
        raw_data = json.load(f)
    
    # Filter out entries with no label
    valid_data = [x for x in raw_data if x.get("period")]
    print(f"📚 Loaded {len(valid_data)} valid records from manifest.")
    return valid_data

# 1. Load Data
try:
    all_data = load_data_from_bridge()
except FileNotFoundError as e:
    print(e)
    exit(1)

# 2. Split Data (80/20)
by_period = defaultdict(list)
for x in all_data:
    by_period[x["period"]].append(x)

train_data, val_data = [], []
periods = []
MIN_SAMPLES = 15  # Threshold to ensure we can split train/val

print("\n📊 Data Distribution:")
for period, items in sorted(by_period.items()):
    if len(items) >= MIN_SAMPLES:
        random.shuffle(items)
        n_val = max(3, int(len(items) * 0.2)) # Ensure at least 3 val samples
        
        val_data.extend(items[:n_val])
        train_data.extend(items[n_val:])
        periods.append(period)
        
        print(f"   {period:<18}: {len(items):5d} items ({len(items)-n_val} train / {n_val} val)")

period_to_idx = {p: i for i, p in enumerate(periods)}
print(f"\n✅ Final: {len(train_data)} Training samples | {len(val_data)} Validation samples")

# 3. Dataset Class
class CoinDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            img = Image.open(item["image_path"]).convert("RGB")
            label = period_to_idx[item["period"]]
            return self.transform(img), label
        except Exception:
            return torch.zeros((3, 128, 128)), 0

# 4. Setup Training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader = DataLoader(CoinDataset(train_data, transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(CoinDataset(val_data, transform), batch_size=BATCH_SIZE, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on {device}...")

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, len(periods))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# 5. Training Loop
best_val = 0
for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct = 0, 0
    
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()
    
    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            val_correct += (model(imgs).argmax(1) == labels).sum().item()
    
    val_acc = val_correct/len(val_data)*100
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "best_model.pth")
    
    print(f"Epoch {epoch+1:2d} | Loss: {train_loss/len(train_loader):.3f} | "
          f"Train: {train_correct/len(train_data)*100:.0f}% | Val: {val_acc:.0f}%")

# 6. Final Evaluation
print(f"\n=== Best Model (Val: {best_val:.0f}%) ===")
model.load_state_dict(torch.load("best_model.pth", weights_only=True))
model.eval()

correct_per_class = defaultdict(int)
total_per_class = defaultdict(int)
confusion = defaultdict(lambda: defaultdict(int))

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        for t, p in zip(labels.tolist(), preds.tolist()):
            total_per_class[periods[t]] += 1
            if t == p:
                correct_per_class[periods[t]] += 1
            confusion[periods[t]][periods[p]] += 1

print("\nPer-class Accuracy:")
for p in periods:
    if total_per_class[p] > 0:
        acc = correct_per_class[p] / total_per_class[p] * 100
        print(f"  {p:18s}: {acc:.0f}% ({correct_per_class[p]}/{total_per_class[p]})")

print("\nConfusion Matrix:")
print(f"{'':18s} " + " ".join(f"{p[:4]:>4s}" for p in periods))
for true_p in periods:
    row = [f"{confusion[true_p][p]:4d}" for p in periods]
    print(f"{true_p:18s}: {' '.join(row)}")
    