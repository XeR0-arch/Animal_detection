# fix_duplicates.py
# Finds and removes duplicate images in your dataset

import os
import hashlib
from collections import defaultdict

print("=" * 55)
print("  🔍 FINDING DUPLICATE IMAGES")
print("=" * 55)

DATASET_DIR = "dataset"

def get_file_hash(filepath):
    """Get unique fingerprint of a file"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)  # Read first 64KB
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

# ========================================
# PHASE 1: Find ALL duplicates
# ========================================
print("\n🔍 Scanning all images...")

hash_to_files = defaultdict(list)
total_files = 0

for split in ["train", "validation"]:
    split_path = os.path.join(DATASET_DIR, split)
    if not os.path.exists(split_path):
        continue
    
    for animal in sorted(os.listdir(split_path)):
        animal_path = os.path.join(split_path, animal)
        if not os.path.isdir(animal_path):
            continue
        
        for img_name in os.listdir(animal_path):
            img_path = os.path.join(animal_path, img_name)
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            total_files += 1
            file_hash = get_file_hash(img_path)
            hash_to_files[file_hash].append(img_path)

print(f"   Total files scanned: {total_files}")

# ========================================
# PHASE 2: Identify duplicates
# ========================================
duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}
duplicate_count = sum(len(files) - 1 for files in duplicates.values())

print(f"   Unique images: {len(hash_to_files)}")
print(f"   Duplicate groups: {len(duplicates)}")
print(f"   Total duplicates to remove: {duplicate_count}")

# Check for cross-contamination (same image in train AND validation)
cross_contamination = 0
for h, files in duplicates.items():
    splits = set()
    for f in files:
        if "train" in f:
            splits.add("train")
        if "validation" in f:
            splits.add("validation")
    if len(splits) > 1:
        cross_contamination += 1

if cross_contamination > 0:
    print(f"\n   ⚠️  CRITICAL: {cross_contamination} images appear in BOTH train AND validation!")
    print(f"   This means your accuracy was FAKE — the model memorized test answers!")
else:
    print(f"\n   ✅ No cross-contamination between train/validation")

# ========================================
# PHASE 3: Show examples
# ========================================
if duplicates:
    print(f"\n📋 Examples of duplicate groups:")
    for i, (h, files) in enumerate(list(duplicates.items())[:5]):
        print(f"\n   Group {i+1} ({len(files)} copies):")
        for f in files:
            print(f"      {f}")

# ========================================
# PHASE 4: Remove duplicates (keep first, delete rest)
# ========================================
if duplicate_count > 0:
    print(f"\n🗑️  Removing {duplicate_count} duplicates...")
    
    removed = 0
    for h, files in duplicates.items():
        # Sort so we prefer keeping train over validation
        # And prefer keeping original over augmented
        files_sorted = sorted(files, key=lambda x: (
            "validation" in x,     # Keep train first
            "aug_" in x,           # Keep originals first
            x                      # Alphabetical
        ))
        
        # Keep the first one, remove the rest
        keep = files_sorted[0]
        for remove_path in files_sorted[1:]:
            os.remove(remove_path)
            removed += 1
    
    print(f"   ✅ Removed {removed} duplicate files")
else:
    print(f"\n   ✅ No duplicates found! Dataset is clean.")

# ========================================
# PHASE 5: Final count
# ========================================
print(f"\n{'='*55}")
print("  📊 CLEANED DATASET SUMMARY")
print(f"{'='*55}")

for split in ["train", "validation"]:
    split_path = os.path.join(DATASET_DIR, split)
    if not os.path.exists(split_path):
        continue
    print(f"\n  {split}/")
    for animal in sorted(os.listdir(split_path)):
        animal_path = os.path.join(split_path, animal)
        if os.path.isdir(animal_path):
            count = len([f for f in os.listdir(animal_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"    {animal:12s}: {count} images")

print(f"\n{'='*55}")

if cross_contamination > 0:
    print("\n  ⚠️  YOU SHOULD RETRAIN THE MODEL!")
    print("     Run: py -3.10 distill.py")
    print("     Your new accuracy will be more REAL this time.")
else:
    print("\n  ✅ Dataset is clean! Model accuracy was legitimate.")