"""Level 2: Roman Emperor Classification"""
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

# Load all data
all_data = []
for split in ["train", "val", "test"]:
    with open(DATA_DIR / f"{split}_manifest.json") as f:
        all_data.extend(json.load(f))

# Filter: Roman Imperial with known authority
roman_data = [x for x in all_data 
              if x.get("period") == "roman_imperial" 
              and x.get("authority")]

by_emperor = defaultdict(list)
for x in roman_data:
    by_emperor[x["authority"]].append(x)

# Keep emperors with 15+ samples
train_data, val_data = [], []
emperors = []
for emperor, items in sorted(by_emperor.items()):
    if len(items) >= 15:
        random.shuffle(items)
        n_val = max(3, len(items) // 5)
        val_data.extend(items[:n_val])
        train_data.extend(items[n_val:])
        emperors.append(emperor)

emperor_to_idx = {e: i for i, e in enumerate(emperors)}

print(f"Emperors ({len(emperors)}): {emperors}")
print(f"Train: {len(train_data)}, Val: {len(val_data)}")
for e in emperors[:10]:
    tc = sum(1 for x in train_data if x["authority"] == e)
    vc = sum(1 for x in val_data if x["authority"] == e)
    print(f"  {e}: {tc} train, {vc} val")
if len(emperors) > 10:
    print(f"  ... and {len(emperors)-10} more")

class CoinDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        return self.transform(img), emperor_to_idx[item["authority"]]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader = DataLoader(CoinDataset(train_data, transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(CoinDataset(val_data, transform), batch_size=BATCH_SIZE, num_workers=0)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, len(emperors))

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
        torch.save(model.state_dict(), "emperor_model.pth")
    
    print(f"Epoch {epoch+1:2d} | Loss: {train_loss/len(train_loader):.3f} | Train: {train_correct/len(train_data)*100:.0f}% | Val: {val_acc:.0f}%")

print(f"\nBest validation accuracy: {best_val:.0f}%")
print(f"Classifying {len(emperors)} emperors")
