"""ML POC with balanced class sampling"""
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
from collections import defaultdict

BATCH_SIZE = 8
EPOCHS = 10
DATA_DIR = Path("trivalaya_data/03_ml_ready/dataset")

def load_manifest(split):
    with open(DATA_DIR / f"{split}_manifest.json") as f:
        return [x for x in json.load(f) if x.get("period")]

# Group by period and balance
def balance_data(data, samples_per_class=50):
    by_period = defaultdict(list)
    for x in data:
        by_period[x["period"]].append(x)
    
    balanced = []
    for period, items in by_period.items():
        if len(items) >= 10:  # Min threshold
            balanced.extend(random.sample(items, min(len(items), samples_per_class)))
    return balanced

train_data = balance_data(load_manifest("train"), samples_per_class=60)
val_data = balance_data(load_manifest("val"), samples_per_class=15)

periods = sorted(set(x["period"] for x in train_data))
period_to_idx = {p: i for i, p in enumerate(periods)}

print(f"Classes: {periods}")
print(f"Train: {len(train_data)}, Val: {len(val_data)}")
for p in periods:
    train_count = sum(1 for x in train_data if x["period"] == p)
    val_count = sum(1 for x in val_data if x["period"] == p)
    print(f"  {p}: {train_count} train, {val_count} val")

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
    
    print(f"Epoch {epoch+1:2d} | Loss: {train_loss/len(train_loader):.3f} | Train: {train_correct/len(train_data)*100:.0f}% | Val: {val_correct/len(val_data)*100:.0f}%")

# Confusion matrix
print("\nConfusion (rows=true, cols=pred):")
confusion = defaultdict(lambda: defaultdict(int))
model.eval()
with torch.no_grad():
    for imgs, labels in val_loader:
        preds = model(imgs).argmax(1)
        for t, p in zip(labels.tolist(), preds.tolist()):
            confusion[periods[t]][periods[p]] += 1

for true_p in periods:
    row = [f"{confusion[true_p][p]:2d}" for p in periods]
    print(f"{true_p:18s}: {' '.join(row)}")

torch.save(model.state_dict(), "coin_classifier_balanced.pth")
print("\nSaved: coin_classifier_balanced.pth")
