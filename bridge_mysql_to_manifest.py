"""
bridge_mysql_to_manifest.py
Connects MySQL auction_data to the ML training pipeline.
Generates train_manifest.json from DB records + human corrections.
"""
import mysql.connector
import json
import os
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DB_CONFIG = {
    "host": os.getenv("TRIVALAYA_DB_HOST", "64.23.235.195"),
    "user": os.getenv("TRIVALAYA_DB_USER", "auction_data"),  # Update with your user
    "password": os.getenv("TRIVALAYA_DB_PASSWORD", "Veritas@2024"), # Update!
    "database": os.getenv("TRIVALAYA_DB_NAME", "auction_data")
}

# Paths
BASE_DIR = Path("trivalaya_data")
CORRECTIONS_DIR = BASE_DIR / "04_corrections"
OUTPUT_DIR = BASE_DIR / "03_ml_ready/dataset"
OUTPUT_PATH = OUTPUT_DIR / "train_manifest.json"

# Label Mapping Rules (Regex/Keyword -> Class)
LABEL_MAP = {
    "roman_imperial": ["imperial", "augustus", "denarius", "sestertius", "antoninianus"],
    "roman_provincial": ["provincial", "cappadocia", "syria", "alexandria", "tetradrachm"],
    "roman_republican": ["republican", "denarius", "anonymous", "crawford"],
    "greek": ["greek", "athens", "tetradrachm", "drachm", "stater", "sicily", "macedon"],
    "byzantine": ["byzantine", "follis", "solidus", "nomisma", "trachy"],
    "islamic": ["islamic", "dirham", "dinar", "fals", "abbasid", "umayyad"],
    "medieval": ["medieval", "penny", "denier", "grosso", "sceat", "styca", "anglo-saxon"],
    "celtic": ["celtic", "stater", "potin", "danubian"],
    "persian": ["persian", "siglos", "daric", "sasanian", "parthian"]
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def infer_label(title, description):
    """Simple keyword matching to guess label from text."""
    text = (str(title) + " " + str(description)).lower()
    
    # Priority checks (e.g. Provincial > Imperial if both present)
    if "provincial" in text: return "roman_provincial"
    if "republic" in text: return "roman_republican"
    
    for label, keywords in LABEL_MAP.items():
        for kw in keywords:
            if kw in text:
                return label
    return None

def main():
    print("🔌 Connecting to MySQL...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
    except Exception as e:
        print(f"❌ DB Connection failed: {e}")
        return

    # 1. Fetch valid records from DB
    print("📥 Fetching records...")
    cursor.execute("""
        SELECT id, lot_number, title, description, image_path, auction_house 
        FROM auction_data 
        WHERE image_path IS NOT NULL AND image_path != ''
    """)
    records = cursor.fetchall()
    
    manifest = []
    stats = defaultdict(int)
    
    # 2. Process DB Records
    for rec in records:
        # Check if file actually exists
        if not os.path.exists(rec['image_path']):
            continue
            
        label = infer_label(rec['title'], rec['description'])
        
        if label:
            manifest.append({
                "image_path": rec['image_path'],
                "period": label,
                "source": "database",
                "db_id": rec['id'],
                "metadata": {
                    "title": rec['title'],
                    "auction_house": rec['auction_house']
                }
            })
            stats[label] += 1

    # 3. Merge Human Corrections (High Priority)
    print("human corrections...")
    if CORRECTIONS_DIR.exists():
        for class_dir in CORRECTIONS_DIR.iterdir():
            if class_dir.is_dir() and class_dir.name in LABEL_MAP:
                label = class_dir.name
                for img_file in class_dir.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                        manifest.append({
                            "image_path": str(img_file),
                            "period": label,
                            "source": "human_correction",
                            "db_id": None
                        })
                        stats[label] += 1

    # 4. Save Manifest
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Created manifest with {len(manifest)} images.")
    print("📊 Class Distribution:")
    for k, v in sorted(stats.items(), key=lambda item: item[1], reverse=True):
        print(f"  {k:<20}: {v}")

    conn.close()

if __name__ == "__main__":
    main()