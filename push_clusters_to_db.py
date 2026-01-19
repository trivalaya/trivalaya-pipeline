import pandas as pd
import mysql.connector
import os
import argparse
from tqdm import tqdm # Optional: pip install tqdm for a progress bar

# CONFIG
DB_HOST = os.getenv("TRIVALAYA_DB_HOST", "127.0.0.1")
DB_USER = os.getenv("TRIVALAYA_DB_USER", "auction_user")
DB_PASSWORD = os.getenv("TRIVALAYA_DB_PASSWORD", "Veritas@2024")
DB_NAME = os.getenv("TRIVALAYA_DB_NAME", "auction_data")
CSV_FILE = 'cluster_output/cluster_results.csv'

def get_db_connection():
    return mysql.connector.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=CSV_FILE, help='Path to cluster results CSV')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"❌ Error: CSV not found at {args.csv}")
        return

    print(f"📂 Loading {args.csv}...")
    df = pd.read_csv(args.csv)
    
    # Ensure ID is a string (handles '16-4', '5-0')
    df['cluster_id'] = df['cluster_id'].astype(str)
    
    # Filter out empty rows just in case
    df = df[df['image_path'].notna()]
    
    print(f"🔌 Connecting to database...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print(f"⚙️  Preparing to update {len(df)} records...")

    # We use executemany for speed. 
    # Query: Match by image_path (most reliable key we have in the CSV)
    update_query = """
        UPDATE ml_dataset 
        SET visual_cluster_id = %s, 
            visual_cluster_name = %s
        WHERE image_path = %s
    """
    
    batch_data = []
    batch_size = 1000
    
    # Iterate and prepare batches
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        c_id = row['cluster_id']
        path = row['image_path']
        
        # Name logic: "Cluster 16-4" or "Noise"
        if c_id == '-1' or 'noise' in c_id.lower():
            c_name = "Noise"
            c_id = "-1" # Standardize noise ID in DB
        else:
            c_name = f"Cluster {c_id}"
            
        batch_data.append((c_id, c_name, path))
        
        # Execute batch
        if len(batch_data) >= batch_size:
            cursor.executemany(update_query, batch_data)
            conn.commit()
            batch_data = []

    # Commit remaining
    if batch_data:
        cursor.executemany(update_query, batch_data)
        conn.commit()

    print(f"✅ Successfully pushed cluster IDs to {DB_NAME}.ml_dataset")
    
    # Verify
    cursor.execute("SELECT COUNT(*) FROM ml_dataset WHERE visual_cluster_id IS NOT NULL")
    count = cursor.fetchone()[0]
    print(f"📊 Total records with cluster info in DB: {count}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()