import mysql.connector
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Copy of your infer_label function to test logic
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

def infer_label(title, description):
    text = (str(title) + " " + str(description)).lower()
    if "provincial" in text: return "roman_provincial"
    if "republic" in text: return "roman_republican"
    for label, keywords in LABEL_MAP.items():
        for kw in keywords:
            if kw in text: return label
    return None

def diagnose():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("TRIVALAYA_DB_HOST", "127.0.0.1"),
            user=os.getenv("TRIVALAYA_DB_USER", "auction_user"),
            password=os.getenv("TRIVALAYA_DB_PASSWORD", "Veritas@2024"),
            database=os.getenv("TRIVALAYA_DB_NAME", "auction_data")
        )
        cursor = conn.cursor(dictionary=True)
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    # 1. Total Records
    cursor.execute("SELECT COUNT(*) as cnt FROM auction_data")
    total = cursor.fetchone()['cnt']
    print(f"1️⃣  Total Rows in MySQL:       {total}")

    # 2. Rows with Image Paths
    cursor.execute("SELECT COUNT(*) as cnt FROM auction_data WHERE image_path IS NOT NULL AND image_path != ''")
    has_path = cursor.fetchone()['cnt']
    print(f"2️⃣  Rows with 'image_path':    {has_path}  (Lost {total - has_path} here)")

    # 3. Test File Existence & Labeling on a sample
    cursor.execute("SELECT id, title, description, image_path FROM auction_data WHERE image_path IS NOT NULL AND image_path != ''")
    records = cursor.fetchall()
    
    missing_files = 0
    unlabeled = 0
    good = 0
    
    print("\n🔍 Analyzing records with paths...")
    for i, rec in enumerate(records):
        # Check File
        if not os.path.exists(rec['image_path']):
            missing_files += 1
            if i < 3: print(f"   ⚠️ Missing File: {rec['image_path']}")
            continue
            
        # Check Label
        label = infer_label(rec['title'], rec['description'])
        if not label:
            unlabeled += 1
            if i < 5: print(f"   ⚠️ Unlabeled: '{rec['title'][:30]}...'")
            continue
            
        good += 1

    print(f"3️⃣  Files Missing on Disk:     {missing_files}")
    print(f"4️⃣  Could not Infer Label:     {unlabeled}")
    print(f"✅  Ready for Training:        {good}")

if __name__ == "__main__":
    diagnose()