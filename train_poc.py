"""Quick ML POC: Period Classification (Low Memory)"""
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from pathlib import Path
from PIL import Image

# Config - reduced for low memory
BATCH_SIZE = 8  # Reduced from 32
EPOCHS = 5
LR = 0.001
DEVICE = "cpu"
DATA_DIR = Path("trivalaya_data/03_ml_ready/dataset")

# Load manifests
def load_manifest(split):
    with open(DATA_DIR / f"{split}_manifest.json") as f:
        return [x for x in json.load(f) if x.get("period")]

train_data = load_manifest("train")
val_data = load_manifest("val")

# Limit data for memory
train_data = train_data[:500]  # Use subset
val_data = val_data[:100]

periods = sorted(set(x["period"] for x in train_data))
period_to_idx = {p: i for i, p in enumerate(periods)}
print(f"Classes ({len(periods)}): {periods}")
print(f"Train: {len(train_data)}, Val: {len(val_data)}")

# Dataset
class CoinDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        label = period_to_idx[item["period"]]
        return self.transform(img), label

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Smaller images
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader = DataLoader(CoinDataset(train_data, transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(CoinDataset(val_data, transform), batch_size=BATCH_SIZE, num_workers=0)

# Smaller model: MobileNetV2
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, len(periods))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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
            outputs = model(imgs)
            val_correct += (outputs.argmax(1) == labels).sum().item()
    
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Loss: {train_loss/len(train_loader):.4f} | "
          f"Train: {train_correct/len(train_data)*100:.1f}% | "
          f"Val: {val_correct/len(val_data)*100:.1f}%")

print("\nDone!")
torch.save(model.state_dict(), "coin_classifier.pth")

# Quick confusion check
from collections import defaultdict
model.eval()
confusion = defaultdict(lambda: defaultdict(int))
with torch.no_grad():
    for imgs, labels in val_loader:
        preds = model(imgs).argmax(1)
        for true, pred in zip(labels.tolist(), preds.tolist()):
            confusion[periods[true]][periods[pred]] += 1

print("\nConfusion (rows=true, cols=pred):")
for true_period in periods:
    row = [f"{confusion[true_period][p]:3d}" for p in periods]
    print(f"{true_period:18s}: {' '.join(row)}")
