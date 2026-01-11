import pandas as pd
import os
import argparse
import sys

def resolve_path(path_str):
    """Try to find the actual image file."""
    candidates = [
        path_str,
        os.path.abspath(path_str),
        os.path.join(os.getcwd(), path_str),
        os.path.join('trivalaya_data', os.path.basename(path_str)),
        os.path.join('../trivalaya_data', os.path.basename(path_str)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, required=True, help='The Cluster ID to split')
    parser.add_argument('--csv', type=str, default='cluster_output/cluster_results.csv')
    args = parser.parse_args()

    # SETUP
    TARGET_CLUSTER = args.target
    SOURCE_CSV = args.csv
    STAGING_DIR = f'cluster_output/cluster_{TARGET_CLUSTER}_staging'

    if not os.path.exists(SOURCE_CSV):
        sys.exit(f"❌ Error: CSV not found at {SOURCE_CSV}")

    print(f"📂 Loading {SOURCE_CSV}...")
    df = pd.read_csv(SOURCE_CSV)
    
    # FILTER
    cluster_col = 'cluster_id' if 'cluster_id' in df.columns else 'cluster_label'
    subset = df[df[cluster_col].astype(str) == str(TARGET_CLUSTER)]
    total_expected = len(subset)
    print(f"🔎 Found {total_expected} images in Cluster {TARGET_CLUSTER}")

    # RESET DIRECTORY (Force Clean Slate)
    if os.path.exists(STAGING_DIR):
        print(f"   🧹 Cleaning existing staging directory...")
        import shutil
        shutil.rmtree(STAGING_DIR)
    os.makedirs(STAGING_DIR)

    count = 0
    missing = 0
    
    print("   Creating collision-proof links...")
    # Iterate with index to ensure uniqueness
    for idx, row in subset.iterrows():
        original_path_str = row['image_path']
        final_path = resolve_path(original_path_str)
        
        if not final_path:
            missing += 1
            continue

        # CRITICAL FIX: Add unique prefix to filename
        # e.g. "coin.jpg" -> "452_coin.jpg"
        filename = os.path.basename(final_path)
        unique_name = f"{idx}_{filename}"
        dst_path = os.path.join(STAGING_DIR, unique_name)
        
        try:
            os.symlink(final_path, dst_path)
            count += 1
        except OSError as e:
            print(f"   ❌ Error linking {filename}: {e}")

    print("-" * 40)
    print(f"✅ Staged: {count}/{total_expected} images")
    print(f"❌ Missing: {missing}")
    print("-" * 40)
    
    if count > 0:
        print(f"🚀 Ready! Run clustering on '{STAGING_DIR}'")

if __name__ == "__main__":
    main()