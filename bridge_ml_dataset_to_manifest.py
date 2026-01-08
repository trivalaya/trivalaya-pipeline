"""
bridge_ml_dataset_to_manifest.py
Reads DIRECTLY from the 'ml_dataset' table shown in your screenshot.
Generates the training manifest for train_final.py.
"""
import mysql.connector
import json
import os
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
DB_CONFIG = {
    "host": os.getenv("TRIVALAYA_DB_HOST", "127.0.0.1"),
    "user": os.getenv("TRIVALAYA_DB_USER", "auction_user"),
    "password": os.getenv("TRIVALAYA_DB_PASSWORD", "Veritas@2024"),
    "database": os.getenv("TRIVALAYA_DB_NAME", "auction_data")
}

# Output location
OUTPUT_DIR = Path("trivalaya_data/03_ml_ready/dataset")
OUTPUT_PATH = OUTPUT_DIR / "train_manifest.json"

# Potential roots to check for image paths
POSSIBLE_ROOTS = [
    Path("."), 
    Path(".."), 
    Path("/root"),
]

def resolve_path(db_path_str):
    """
    The DB has paths like 'trivalaya_data/03_ml_ready/dataset/train/...'
    We need to find where that actually lives relative to this script.
    """
    if not db_path_str: return None
    
    # Check if the path is absolute or exists as-is
    p = Path(db_path_str)
    if p.exists():
        return str(p.resolve())

    # Check relative to possible roots
    for root in POSSIBLE_ROOTS:
        candidate = root / p
        if candidate.exists():
            return str(candidate.resolve())
            
    return None

def main():
    print("🔌 Connecting to MySQL table: ml_dataset...")
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        # Query based on your screenshot columns
        query = """
            SELECT id, image_path, period, is_verified 
            FROM ml_dataset 
            WHERE image_path IS NOT NULL AND period IS NOT NULL
        """
        cursor.execute(query)
        records = cursor.fetchall()
        
    except Exception as e:
        print(f"❌ DB Error: {e}")
        return

    manifest = []
    stats = defaultdict(int)
    found_count = 0
    lost_count = 0

    print(f"📥 Processing {len(records)} records from ml_dataset...")

    for rec in records:
        # 1. Resolve File Path
        real_path = resolve_path(rec['image_path'])
        
        if not real_path:
            lost_count += 1
            # Print first few errors to help debug
            if lost_count <= 3:
                print(f"   ⚠️ File not found: {rec['image_path']}")
            continue
            
        found_count += 1
        label = rec['period']
        
        # 2. Add to Manifest
        manifest.append({
            "image_path": real_path,
            "period": label,
            "source": "ml_dataset",
            "db_id": rec['id'],
            "is_verified": bool(rec['is_verified'])
        })
        stats[label] += 1

    # 3. Save Manifest
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("-" * 30)
    print(f"✅ Images Verified & Linked: {found_count}")
    print(f"❌ Images Missing on Disk:  {lost_count}")
    print("-" * 30)
    print("Class Distribution:")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k:<20}: {v}")

if __name__ == "__main__":
    main()