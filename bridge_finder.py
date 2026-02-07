import json
import os

print("🔍 DIAGNOSTIC: Checking for Overlap between Manifest and Embeddings...")

# 1. Load Manifest (The "Leu 1" data)
with open('trivalaya_data/03_ml_ready/dataset/all_pairs_manifest.regenerated.json') as f:
    man = json.load(f)
    if not man:
        print("❌ Manifest is empty!"); exit()
    
    # Get a sample ID and Path
    sample = man[0]
    man_id = str(sample.get('coin_id', 'UNKNOWN'))
    man_path = sample.get('obv_path', '')
    print(f"\n📄 Manifest Sample:")
    print(f"   - Coin ID: {man_id}")
    print(f"   - Path:    {man_path}")

# 2. Load Embeddings Metadata (The "Leu 67" data?)
with open('trivalaya_embeddings_meta.json') as f:
    meta = json.load(f)
    emb_paths = meta.get('paths', [])
    emb_ids = meta.get('coin_ids', []) # Check if this list exists
    
    print(f"\n📦 Embeddings Metadata:")
    print(f"   - Total Paths: {len(emb_paths)}")
    print(f"   - Total IDs:   {len(emb_ids)}")
    if len(emb_paths) > 0:
        print(f"   - Sample Path: {emb_paths[0]}")

# 3. TEST 1: Direct ID Match
print(f"\n🧪 TEST 1: Searching for ID '{man_id}' in Embeddings...")
if str(man_id) in [str(x) for x in emb_ids]:
    print("   ✅ SUCCESS! ID found in 'coin_ids' list.")
else:
    print("   ❌ FAILED. ID not found in embeddings metadata.")

# 4. TEST 2: Path/Filename Match
# Extract strictly the filename (e.g. "Lot_01006_obv_224.jpg")
man_filename = os.path.basename(man_path)
print(f"\n🧪 TEST 2: Searching for filename '{man_filename}'...")

found_filename = False
for p in emb_paths:
    if man_filename in p:
        found_filename = True
        print(f"   ✅ SUCCESS! Found match: {p}")
        break

if not found_filename:
    print("   ❌ FAILED. Filename not found in embeddings.")
    
# 5. TEST 3: Broad Auction Match (Leu 1)
print(f"\n🧪 TEST 3: checking if ANY 'leu_1' or 'Leu 1' data exists in embeddings...")
leu1_matches = [p for p in emb_paths if "leu_1" in p.lower() or "leu/1" in p.lower()]
if leu1_matches:
    print(f"   ✅ YES! Found {len(leu1_matches)} paths containing 'leu_1'.")
else:
    print("   ❌ NO. Your embeddings file does not seem to contain any Leu 1 data.")
    print("      It mostly contains paths like:", emb_paths[0] if emb_paths else "N/A")