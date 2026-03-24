# expand_dataset.py
# Expands dataset: adds new animals + augments to 300+ images per class

import os
import shutil
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import math

# ========================================
# CONFIG
# ========================================

# Where the Kaggle 90-animals dataset was unzipped
# The script will search for it if this path is wrong
SOURCE_DIR = os.path.join("archive", "animals", "animals")

# Animals you ALREADY have
EXISTING_ANIMALS = ["elephant", "leopard", "lion", "zebra"]

# NEW animals to add (common jungle/safari animals)
NEW_ANIMALS = ["bear", "deer", "gorilla", "tiger"]

# All animals combined
ALL_ANIMALS = EXISTING_ANIMALS + NEW_ANIMALS

# Target: how many images per class MINIMUM
TARGET_PER_CLASS = 300

# Train/validation split
TRAIN_RATIO = 0.8

print("=" * 55)
print("  🌍 EXPANDING JUNGLE ANIMAL DATASET")
print("=" * 55)

# ========================================
# PHASE 1: Find the source dataset
# ========================================
print("\n📂 Phase 1: Finding source dataset...")

if not os.path.exists(SOURCE_DIR):
    print(f"   ❌ Can't find: {SOURCE_DIR}")
    print(f"   🔍 Searching for animal folders...")
    
    found = False
    for root, dirs, files in os.walk("."):
        if "elephant" in dirs and "lion" in dirs:
            SOURCE_DIR = root
            print(f"   ✅ Found at: {SOURCE_DIR}")
            found = True
            break
    
    if not found:
        print("   ❌ Could not find the Kaggle animal folders!")
        print("   Make sure the 90-animals dataset is unzipped here.")
        exit()
else:
    print(f"   ✅ Found: {SOURCE_DIR}")

# Show available animals
available = sorted([d for d in os.listdir(SOURCE_DIR) 
                   if os.path.isdir(os.path.join(SOURCE_DIR, d))])
print(f"   📋 {len(available)} animal folders available")

# ========================================
# PHASE 2: Copy NEW animals from source
# ========================================
print("\n📂 Phase 2: Adding new animals...")

# Create a temporary "all_raw" folder with everything
RAW_DIR = "dataset_raw"
os.makedirs(RAW_DIR, exist_ok=True)

for animal in ALL_ANIMALS:
    raw_animal_dir = os.path.join(RAW_DIR, animal)
    
    # Check if we already have images in dataset/train
    existing_train = os.path.join("dataset", "train", animal)
    existing_val = os.path.join("dataset", "validation", animal)
    
    os.makedirs(raw_animal_dir, exist_ok=True)
    
    # Copy from existing dataset (train + validation) if exists
    for existing_dir in [existing_train, existing_val]:
        if os.path.exists(existing_dir):
            for f in os.listdir(existing_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    src = os.path.join(existing_dir, f)
                    dst = os.path.join(raw_animal_dir, f"existing_{f}")
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
    
    # Copy from Kaggle source
    # Try exact name first, then search for partial match
    source_animal_dir = os.path.join(SOURCE_DIR, animal)
    
    if not os.path.exists(source_animal_dir):
        # Search for partial match (e.g., "cheetah" might be "Cheetah")
        matches = [a for a in available if animal.lower() in a.lower()]
        if matches:
            source_animal_dir = os.path.join(SOURCE_DIR, matches[0])
            print(f"   🔍 '{animal}' → found as '{matches[0]}'")
        else:
            print(f"   ⚠️  '{animal}' not found in Kaggle dataset!")
            print(f"      Close matches: {[a for a in available if a[0].lower() == animal[0].lower()]}")
            continue
    
    if os.path.exists(source_animal_dir):
        count = 0
        for f in os.listdir(source_animal_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                src = os.path.join(source_animal_dir, f)
                dst = os.path.join(raw_animal_dir, f"kaggle_{f}")
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    count += 1
        print(f"   ✅ {animal}: copied {count} images from Kaggle")

# Count raw images
print("\n   📊 Raw image counts:")
for animal in ALL_ANIMALS:
    raw_dir = os.path.join(RAW_DIR, animal)
    if os.path.exists(raw_dir):
        count = len([f for f in os.listdir(raw_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"      {animal:12s}: {count} images")

# ========================================
# PHASE 3: Validate images (remove broken ones)
# ========================================
print("\n🔍 Phase 3: Validating images (removing broken ones)...")

removed = 0
for animal in ALL_ANIMALS:
    raw_dir = os.path.join(RAW_DIR, animal)
    if not os.path.exists(raw_dir):
        continue
    
    for f in os.listdir(raw_dir):
        filepath = os.path.join(raw_dir, f)
        try:
            with Image.open(filepath) as img:
                img.verify()
            # Double check by actually loading it
            with Image.open(filepath) as img:
                img = img.convert("RGB")
                w, h = img.size
                if w < 10 or h < 10:  # Too small
                    os.remove(filepath)
                    removed += 1
        except Exception:
            os.remove(filepath)
            removed += 1

print(f"   🗑️  Removed {removed} broken/invalid images")

# ========================================
# PHASE 4: Augment images to reach target count
# ========================================
print(f"\n🎨 Phase 4: Augmenting to {TARGET_PER_CLASS}+ images per class...")
print("   (Creating flipped, rotated, color-shifted copies)")

def create_augmented_image(img, aug_type):
    """Create one augmented version of an image"""
    
    if aug_type == 0:
        # Horizontal flip
        return ImageOps.mirror(img)
    
    elif aug_type == 1:
        # Random rotation (-15 to +15 degrees)
        angle = random.uniform(-15, 15)
        return img.rotate(angle, fillcolor=(0, 0, 0), expand=False)
    
    elif aug_type == 2:
        # Brightness change
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.6, 1.4)
        return enhancer.enhance(factor)
    
    elif aug_type == 3:
        # Contrast change
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.6, 1.4)
        return enhancer.enhance(factor)
    
    elif aug_type == 4:
        # Flip + brightness
        img = ImageOps.mirror(img)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(random.uniform(0.7, 1.3))
    
    elif aug_type == 5:
        # Rotation + contrast
        angle = random.uniform(-20, 20)
        img = img.rotate(angle, fillcolor=(0, 0, 0))
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(random.uniform(0.7, 1.3))
    
    elif aug_type == 6:
        # Color shift
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(random.uniform(0.7, 1.3))
    
    elif aug_type == 7:
        # Slight blur (simulates camera motion)
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    
    elif aug_type == 8:
        # Flip + rotation + color
        img = ImageOps.mirror(img)
        img = img.rotate(random.uniform(-10, 10), fillcolor=(0, 0, 0))
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(random.uniform(0.8, 1.2))
    
    else:
        # Sharpness change
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(random.uniform(0.5, 2.0))


for animal in ALL_ANIMALS:
    raw_dir = os.path.join(RAW_DIR, animal)
    if not os.path.exists(raw_dir):
        print(f"   ⚠️  Skipping {animal} (no images)")
        continue
    
    # Get current images
    original_images = [
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        and not f.startswith("aug_")  # Don't augment augmented images
    ]
    
    current_count = len(original_images)
    
    if current_count == 0:
        print(f"   ❌ {animal}: no original images to augment!")
        continue
    
    if current_count >= TARGET_PER_CLASS:
        print(f"   ✅ {animal}: already has {current_count} images (target: {TARGET_PER_CLASS})")
        continue
    
    # How many augmented images do we need?
    needed = TARGET_PER_CLASS - current_count
    
    print(f"   🎨 {animal}: {current_count} originals → creating {needed} augmented copies...")
    
    aug_count = 0
    aug_type = 0
    
    while aug_count < needed:
        for img_name in original_images:
            if aug_count >= needed:
                break
            
            img_path = os.path.join(raw_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img = img.resize((96, 96))  # Resize to our model's input
                    
                    # Create augmented version
                    aug_img = create_augmented_image(img, aug_type % 10)
                    
                    # Save
                    aug_name = f"aug_{aug_type}_{aug_count}_{img_name}"
                    # Make sure it's .jpg
                    if not aug_name.lower().endswith('.jpg'):
                        aug_name = aug_name.rsplit('.', 1)[0] + '.jpg'
                    
                    aug_path = os.path.join(raw_dir, aug_name)
                    aug_img.save(aug_path, "JPEG", quality=90)
                    aug_count += 1
                    
            except Exception as e:
                continue
        
        aug_type += 1
        
        # Safety: don't infinite loop
        if aug_type > 100:
            print(f"      ⚠️  Could only create {aug_count}/{needed} augmented images")
            break
    
    total = len([f for f in os.listdir(raw_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"      → {animal} now has {total} images total ✅")

# ========================================
# PHASE 5: Split into train/validation
# ========================================
print(f"\n📂 Phase 5: Splitting into train ({int(TRAIN_RATIO*100)}%) / validation ({int((1-TRAIN_RATIO)*100)}%)...")

# Clear old dataset
if os.path.exists("dataset"):
    shutil.rmtree("dataset")
    print("   🗑️  Cleared old dataset/")

for animal in ALL_ANIMALS:
    raw_dir = os.path.join(RAW_DIR, animal)
    if not os.path.exists(raw_dir):
        continue
    
    # Get all images
    all_images = [
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    if len(all_images) == 0:
        print(f"   ⚠️  {animal}: no images, skipping!")
        continue
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_images)
    
    # Split
    split_idx = int(len(all_images) * TRAIN_RATIO)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Copy to final dataset folders
    for split_name, image_list in [("train", train_images), ("validation", val_images)]:
        dst_dir = os.path.join("dataset", split_name, animal)
        os.makedirs(dst_dir, exist_ok=True)
        
        for img_name in image_list:
            src = os.path.join(raw_dir, img_name)
            dst = os.path.join(dst_dir, img_name)
            shutil.copy2(src, dst)

# ========================================
# PHASE 6: Final Summary
# ========================================
print(f"\n{'=' * 55}")
print("  📊 FINAL DATASET SUMMARY")
print(f"{'=' * 55}")

grand_total_train = 0
grand_total_val = 0

for split in ["train", "validation"]:
    print(f"\n  {split}/")
    split_path = os.path.join("dataset", split)
    if os.path.exists(split_path):
        for animal in sorted(os.listdir(split_path)):
            animal_path = os.path.join(split_path, animal)
            if os.path.isdir(animal_path):
                count = len(os.listdir(animal_path))
                bar = "█" * (count // 10)
                print(f"    {animal:12s}  {count:4d} images  {bar}")
                if split == "train":
                    grand_total_train += count
                else:
                    grand_total_val += count

print(f"\n  ────────────────────────────────────")
print(f"  Total Training:    {grand_total_train} images")
print(f"  Total Validation:  {grand_total_val} images")
print(f"  Grand Total:       {grand_total_train + grand_total_val} images")
print(f"  Classes:           {len(ALL_ANIMALS)} animals")
print(f"  Animals:           {ALL_ANIMALS}")
print(f"{'=' * 55}")

print(f"\n  ✅ データセット準備完了！(Dataset ready!)")
print(f"\n  Next step: retrain the model!")
print(f"  Run: py -3.10 train.py")

# Cleanup hint
print(f"\n  💡 Tip: You can delete 'dataset_raw/' folder later")
print(f"     to save disk space (~{grand_total_train + grand_total_val} images × 2)")