import torch
import numpy as np
import json
from PIL import Image
from pathlib import Path
import os

# --- USE OPEN_CLIP (Match your original script) ---
import open_clip 

# Config
BATCH_SIZE = 32  # Lower batch size for 2GB RAM
OUTPUT_DIR = Path("cluster_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Database connection (Use your existing logic)
DB_HOST = os.getenv("TRIVALAYA_DB_HOST", "127.0.0.1")
DB_USER = os.getenv("TRIVALAYA_DB_USER", "auction_user")
DB_PASSWORD = os.getenv("TRIVALAYA_DB_PASSWORD", "Veritas@2024")
DB_NAME = os.getenv("TRIVALAYA_DB_NAME", "auction_data")

def get_db_connection():
    import mysql.connector
    return mysql.connector.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
    )

def load_dataset_from_db():
    print("📊 Loading dataset...")
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = "SELECT id, image_path FROM ml_dataset WHERE image_path IS NOT NULL AND is_active = 1"
    cursor.execute(query)
    records = cursor.fetchall()
    conn.close()
    return records

def resolve_image_path(db_path):
    # Simple resolver - adjust if your paths are complex
    p = Path(db_path)
    if p.exists(): return p
    # Try relative to current dir
    p = Path(".") / db_path
    if p.exists(): return p
    return None

# --- MAIN ---
device = "cpu" # Force CPU for 2GB RAM server (GPU will crash if shared mem is low)
print(f"🧠 Loading OpenCLIP model on {device}...")

# Using the exact same model config from your original script
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()

records = load_dataset_from_db()
paths = [r['image_path'] for r in records]

features = []
valid_paths = []

print(f"🔍 Extracting features for {len(paths)} images...")

with torch.no_grad():
    for i in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[i:i + BATCH_SIZE]
        batch_imgs = []
        current_batch_indices = []
        
        for idx, p in enumerate(batch_paths):
            path = resolve_image_path(p)
            if not path: continue
            
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = preprocess(img)
                batch_imgs.append(img_tensor)
                current_batch_indices.append(idx)
            except Exception:
                continue
        
        if batch_imgs:
            batch_tensor = torch.stack(batch_imgs)
            # OpenCLIP syntax for encoding
            batch_features = model.encode_image(batch_tensor)
            
            # Normalize
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
            
            features.append(batch_features.cpu().numpy())
            valid_paths.extend([batch_paths[x] for x in current_batch_indices])
            
        if i % 100 == 0:
            print(f"  Processed {i}/{len(paths)}...")

# Save
if features:
    all_features = np.concatenate(features)
    np.save(OUTPUT_DIR / "trivalaya_clip_embeddings.npy", all_features)
    
    with open(OUTPUT_DIR / "trivalaya_clip_meta.json", "w") as f:
        json.dump({"paths": valid_paths}, f)

    print(f"✅ Saved {len(all_features)} embeddings to {OUTPUT_DIR}")
else:
    print("❌ No features extracted.")