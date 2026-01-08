import mysql.connector
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- REUSE CONFIG FROM BRIDGE SCRIPT ---
POSSIBLE_IMAGE_ROOTS = [
    Path("."), Path(".."), Path("../data_code"), 
    Path("trivalaya_data/01_raw"), Path("/root/trivalaya_data")
]

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

def resolve_path(db_path):
    for root in POSSIBLE_IMAGE_ROOTS:
        candidate = root / db_path
        if candidate.exists(): return True
    return False

def infer_label(title, description):
    text = (str(title) + " " + str(description)).lower()
    if "provincial" in text: return "roman_provincial"
    if "republic" in text: return "roman_republican"
    for label, keywords in LABEL_MAP.items():
        for kw in keywords:
            if kw in text: return label
    return None

def main():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("TRIVALAYA_DB_HOST", "127.0.0.1"),
            user=os.getenv("TRIVALAYA_DB_USER", "auction_user"),
            password=os.getenv("TRIVALAYA_DB_PASSWORD", "Veritas@2024"),
            database=os.getenv("TRIVALAYA_DB_NAME", "auction_data")
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, title, description, image_path FROM auction_data WHERE image_path IS NOT NULL AND image_path != ''")
        records = cursor.fetchall()
    except Exception as e:
        print(f"❌ DB Error: {e}")
        return

    missing_files = []
    unlabeled = []
    
    print(f"🔍 Auditing {len(records)} records...")

    for rec in records:
        # 1. Check File
        if not resolve_path(rec['image_path']):
            missing_files.append(rec)
            continue
            
        # 2. Check Label
        if not infer_label(rec['title'], rec['description']):
            unlabeled.append(rec)

    print("\n" + "="*50)
    print(f"❌ RECORDS EXCLUDED: {len(missing_files) + len(unlabeled)}")
    print("="*50)

    if missing_files:
        print(f"\n📁 MISSING FILES ({len(missing_files)}):")
        print("   (Records exist in DB, but image file not found)")
        for r in missing_files[:5]:
            print(f"   - [ID {r['id']}] Path: {r['image_path']}")

    if unlabeled:
        print(f"\n🏷️  UNLABELED ({len(unlabeled)}):")
        print("   (Image exists, but description didn't match any class keywords)")
        for r in unlabeled[:10]:
            print(f"   - [ID {r['id']}] {r['title'][:60]}...")
            # print(f"     Desc: {r['description'][:50]}...") # Uncomment to see desc

    print("\n✅ To fix 'Unlabeled': Add their keywords to LABEL_MAP in bridge_mysql_to_manifest.py")

if __name__ == "__main__":
    main()