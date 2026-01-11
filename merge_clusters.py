import pandas as pd
import shutil
import argparse
import os
import sys
import re
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--master', default='cluster_output/cluster_results.csv', help='Main cluster_results.csv')
    parser.add_argument('--sub', required=True, help='Sub-cluster results CSV')
    parser.add_argument('--parent', required=True, type=int, help='Parent Cluster ID (e.g., 5)')
    args = parser.parse_args()

    # 1. Validation
    if not os.path.exists(args.master):
        sys.exit(f"❌ Master CSV not found: {args.master}")
    if not os.path.exists(args.sub):
        sys.exit(f"❌ Sub-cluster CSV not found: {args.sub}")

    # 2. Backup
    timestamp = datetime.now().strftime("%H%M%S")
    backup = f"{args.master}.{timestamp}.bak"
    shutil.copyfile(args.master, backup)
    print(f"🔒 Backed up master to {backup}")

    # 3. Load Data
    print(f"📂 Loading master: {args.master}")
    master_df = pd.read_csv(args.master)
    # Ensure ID is string to allow "5-0"
    master_df['cluster_id'] = master_df['cluster_id'].astype(str)
    
    print(f"📂 Loading sub-results: {args.sub}")
    sub_df = pd.read_csv(args.sub)

    # 4. Create Mapping (Original Index -> New ID)
    # The collision-proof splitter prefixed files with "INDEX_filename".
    # We can trust that INDEX to map back to the master DF directly if we are careful,
    # OR we can just strip the prefix and match by filename (safer if indices shifted).
    
    new_map = {}
    
    print("   Building map...")
    for _, row in sub_df.iterrows():
        # Sub-cluster filename: "452_coin.jpg"
        sub_filename = os.path.basename(row['image_path'])
        
        # Regex to strip the index prefix: "123_name.jpg" -> "name.jpg"
        # We look for digits followed by an underscore at the start
        match = re.match(r"^\d+_(.+)$", sub_filename)
        
        if match:
            original_filename = match.group(1)
        else:
            # Fallback if no prefix found (unlikely)
            original_filename = sub_filename

        # Construct new ID
        if row['cluster_id'] == -1:
            new_id = f"{args.parent}-noise"
        else:
            new_id = f"{args.parent}-{row['cluster_id']}"
            
        new_map[original_filename] = new_id

    # 5. Update Master
    print(f"🔄 Merging {len(new_map)} new assignments...")
    updates = 0
    
    for idx, row in master_df.iterrows():
        current_id = str(row['cluster_id'])
        
        # Optimization: Only look at rows currently in the parent cluster
        if current_id == str(args.parent):
            # Get the master filename
            master_fname = os.path.basename(row['image_path'])
            
            if master_fname in new_map:
                new_val = new_map[master_fname]
                master_df.at[idx, 'cluster_id'] = new_val
                master_df.at[idx, 'visual_cluster'] = f"cluster_{new_val}"
                updates += 1

    # 6. Save
    master_df.to_csv(args.master, index=False)
    print(f"✅ Success! Updated {updates} records in {args.master}")
    print(f"   Cluster {args.parent} has been replaced with sub-clusters (e.g., {args.parent}-0, {args.parent}-1)")

if __name__ == "__main__":
    main()