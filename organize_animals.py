# organize_data.py
# Picks only jungle animals and splits into train/validation

import os
import shutil
import random

# ========================================
# CHANGE THIS to match where you unzipped!
# Look inside your unzipped folder for the 
# individual animal folders (elephant/, lion/, etc.)
# ========================================
SOURCE_DIR = os.path.join("archive", "animals", "animals")

# ========================================
# Pick which jungle animals you want
# Start with 4 animals (you can add more later)
# The folder names must match EXACTLY (lowercase)
# ========================================
ANIMALS_I_WANT = [
    "elephant",
    "lion",
    "leopard",
    "zebra",
]

# Train/validation split ratio
TRAIN_RATIO = 0.8   # 80% training, 20% validation

# ========================================
# Check source exists
# ========================================
if not os.path.exists(SOURCE_DIR):
    print(f"❌ Can't find: {SOURCE_DIR}")
    print(f"\nLet me search for the animal folders...")
    
    # Try to find them
    for root, dirs, files in os.walk("."):
        if "elephant" in dirs and "lion" in dirs:
            print(f"✅ Found them at: {root}")
            print(f"\nChange SOURCE_DIR in this script to:")
            print(f'   SOURCE_DIR = r"{root}"')
            break
    else:
        print("❌ Could not find animal folders.")
        print("Make sure you unzipped the Kaggle download in this folder.")
    exit()

# ========================================
# Organize!
# ========================================
print("=" * 50)
print("  Organizing Jungle Animal Dataset")
print("=" * 50)

for animal in ANIMALS_I_WANT:
    src_folder = os.path.join(SOURCE_DIR, animal)
    
    if not os.path.exists(src_folder):
        print(f"\n❌ '{animal}' folder not found at {src_folder}")
        print(f"   Available folders:")
        available = sorted(os.listdir(SOURCE_DIR))
        # Show folders that partially match
        matches = [a for a in available if animal[:3] in a.lower()]
        if matches:
            print(f"   Did you mean: {matches}?")
        else:
            print(f"   {available[:20]}...")  # Show first 20
        continue
    
    # Get all image files
    all_images = [
        f for f in os.listdir(src_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ]
    
    if len(all_images) == 0:
        print(f"\n❌ No images found in {src_folder}")
        continue
    
    # Shuffle randomly
    random.seed(42)  # For reproducibility
    random.shuffle(all_images)
    
    # Split into train and validation
    split_idx = int(len(all_images) * TRAIN_RATIO)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Copy to organized folders
    for split_name, image_list in [("train", train_images), ("validation", val_images)]:
        dst_folder = os.path.join("dataset", split_name, animal)
        os.makedirs(dst_folder, exist_ok=True)
        
        for img_name in image_list:
            src_path = os.path.join(src_folder, img_name)
            dst_path = os.path.join(dst_folder, img_name)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
    
    print(f"\n✅ {animal}:")
    print(f"   Train:      {len(train_images)} images")
    print(f"   Validation: {len(val_images)} images")

# ========================================
# Final Summary
# ========================================
print(f"\n{'=' * 50}")
print("  📊 FINAL DATASET SUMMARY")
print(f"{'=' * 50}")

total_train = 0
total_val = 0

for split in ["train", "validation"]:
    split_path = os.path.join("dataset", split)
    if not os.path.exists(split_path):
        continue
    print(f"\n  {split}/")
    for animal in sorted(os.listdir(split_path)):
        animal_path = os.path.join(split_path, animal)
        if os.path.isdir(animal_path):
            count = len(os.listdir(animal_path))
            print(f"    └── {animal:12s}  → {count} images")
            if split == "train":
                total_train += count
            else:
                total_val += count

print(f"\n  Total: {total_train} training + {total_val} validation")
print(f"\n{'=' * 50}")
print("  ✅ Dataset organized and ready!")
print(f"{'=' * 50}")

# Warn if too few images
if total_train < 100:
    print("\n⚠️  WARNING: Very few images! Model might not learn well.")
    print("   Try to have at least 60-80 images per animal.")

print(f"\nYour folder now looks like:")
print(f"  dataset/")
print(f"  ├── train/")
for a in ANIMALS_I_WANT:
    print(f"  │   ├── {a}/")
print(f"  └── validation/")
for a in ANIMALS_I_WANT:
    print(f"      ├── {a}/")