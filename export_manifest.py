import mysql.connector
import json
import os
from pathlib import Path

# CONFIG
DB_HOST = os.getenv("TRIVALAYA_DB_HOST", "127.0.0.1")
DB_USER = os.getenv("TRIVALAYA_DB_USER", "auction_user")
DB_PASSWORD = os.getenv("TRIVALAYA_DB_PASSWORD", "Veritas@2024")
DB_NAME = os.getenv("TRIVALAYA_DB_NAME", "auction_data")
OUTPUT_DIR = Path("trivalaya_data/03_ml_ready/dataset")

def get_db_connection():
    return mysql.connector.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
    )

def main():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    print("🔌 exporting active records from DB...")
    
    # Select ONLY active records with valid periods
    query = """
        SELECT id, image_path, period, raw_label, visual_cluster_id
        FROM ml_dataset 
        WHERE is_active = 1 
        AND period IS NOT NULL 
        AND period != ''
        AND image_path IS NOT NULL
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    # Ensure output dir exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save
    outfile = OUTPUT_DIR / "train_manifest.json"
    with open(outfile, 'w') as f:
        json.dump(rows, f, indent=2)
        
    print(f"✅ Exported {len(rows)} records to {outfile}")
    print(f"   Ready for training script.")

if __name__ == "__main__":
    main()