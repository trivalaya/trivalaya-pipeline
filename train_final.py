"""ML POC with proper balanced train/val"""
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
from collections import defaultdict

random.seed(42)
BATCH_SIZE = 8
EPOCHS = 15
DATA_DIR = Path("trivalaya_data/03_ml_ready/dataset")

def load_all():
    all_data = []
    for split in ["train", "val", "test"]:
        with open(DATA_DIR / f"{split}_manifest.json") as f:
            all_data.extend([x for x in json.load(f) if x.get("period")])
    return all_data

# Load all data and split ourselves
all_data = load_all()
by_period = defaultdict(list)
for x in all_data:
    by_period[x["period"]].append(x)

# Only keep periods with enough samples
train_data, val_data = [], []
periods = []
for period, items in sorted(by_period.items()):
    if len(items) >= 20:  # Need at least 20 samples
        random.shuffle(items)
        n_val = max(5, len(items) // 5)  # 20% for val, min 5
        val_data.extend(items[:n_val])
        train_data.extend(items[n_val:n_val+60])  # Cap at 60 train
        periods.append(period)

period_to_idx = {p: i for i, p in enumerate(periods)}

print(f"Classes: {periods}")
for p in periods:
    tc = sum(1 for x in train_data if x["period"] == p)
    vc = sum(1 for x in val_data if x["period"] == p)
    print(f"  {p}: {tc} train, {vc} val")
print(f"Total: {len(train_data)} train, {len(val_data)} val")

class CoinDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        return self.transform(img), period_to_idx[item["period"]]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader = DataLoader(CoinDataset(train_data, transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(CoinDataset(val_data, transform), batch_size=BATCH_SIZE, num_workers=0)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, len(periods))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

best_val = 0
for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct = 0, 0
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()
    
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            val_correct += (model(imgs).argmax(1) == labels).sum().item()
    
    val_acc = val_correct/len(val_data)*100
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "best_model.pth")
    
    print(f"Epoch {epoch+1:2d} | Loss: {train_loss/len(train_loader):.3f} | Train: {train_correct/len(train_data)*100:.0f}% | Val: {val_acc:.0f}%")

# Load best model for final eval
model.load_state_dict(torch.load("best_model.pth", weights_only=True))
model.eval()

print(f"\n=== Best Model (Val: {best_val:.0f}%) ===")
print("\nPer-class accuracy:")
correct_per_class = defaultdict(int)
total_per_class = defaultdict(int)
confusion = defaultdict(lambda: defaultdict(int))

with torch.no_grad():
    for imgs, labels in val_loader:
        preds = model(imgs).argmax(1)
        for t, p in zip(labels.tolist(), preds.tolist()):
            total_per_class[periods[t]] += 1
            if t == p:
                correct_per_class[periods[t]] += 1
            confusion[periods[t]][periods[p]] += 1

for p in periods:
    acc = correct_per_class[p] / total_per_class[p] * 100 if total_per_class[p] > 0 else 0
    print(f"  {p:18s}: {correct_per_class[p]:2d}/{total_per_class[p]:2d} = {acc:.0f}%")

print("\nConfusion matrix:")
print(f"{'':18s}  " + " ".join(f"{p[:4]:>4s}" for p in periods))
for true_p in periods:
    row = [f"{confusion[true_p][p]:4d}" for p in periods]
    print(f"{true_p:18s}: {' '.join(row)}")
