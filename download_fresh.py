# download_fresh.py
# Downloads 1000+ FRESH images per animal from Bing
# One animal at a time, validated and deduplicated

import os
import shutil
import hashlib
import random
from PIL import Image
from icrawler.builtin import BingImageCrawler

# ========================================
# CONFIG
# ========================================
ANIMALS = {
    "bear": [
        "bear wildlife photography",
        "brown bear in forest",
        "black bear jungle",
        "bear in wild nature",
        "grizzly bear wildlife",
        "bear walking forest trail",
    ],
    "deer": [
        "deer wildlife photography",
        "deer in forest",
        "wild deer jungle",
        "spotted deer nature",
        "deer safari photography",
        "deer standing alert wild",
    ],
    "elephant": [
        "elephant wildlife photography",
        "african elephant safari",
        "elephant in jungle",
        "wild elephant nature",
        "elephant herd wildlife",
        "indian elephant forest",
    ],
    "gorilla": [
        "gorilla wildlife photography",
        "gorilla in jungle",
        "silverback gorilla wild",
        "mountain gorilla nature",
        "gorilla forest photography",
        "gorilla sitting wild",
    ],
    "leopard": [
        "leopard wildlife photography",
        "leopard in jungle",
        "leopard safari photo",
        "leopard on tree wild",
        "leopard hunting wildlife",
        "indian leopard forest",
    ],
    "lion": [
        "lion wildlife photography",
        "lion safari africa",
        "lion in wild grass",
        "male lion nature photo",
        "lion pride wildlife",
        "lion resting savanna",
    ],
    "tiger": [
        "tiger wildlife photography",
        "bengal tiger jungle",
        "tiger in forest wild",
        "tiger safari photography",
        "siberian tiger nature",
        "tiger walking wild trail",
    ],
    "zebra": [
        "zebra wildlife photography",
        "zebra safari africa",
        "zebra in wild grass",
        "zebra herd nature",
        "zebra plains wildlife",
        "zebra drinking water wild",
    ],
}

IMAGES_PER_QUERY = 200        # Download 200 per search query
TARGET_PER_ANIMAL = 1000      # Want 1000 per animal
TRAIN_RATIO = 0.85
RAW_DIR = "raw_downloads"
FINAL_DIR = "dataset"

print("=" * 60)
print("  🌍 FRESH DATASET DOWNLOADER")
print(f"  Animals: {list(ANIMALS.keys())}")
print(f"  Target: {TARGET_PER_ANIMAL} images per animal")
print("=" * 60)

# ========================================
# PHASE 1: Download images for each animal
# ========================================
for animal_idx, (animal, queries) in enumerate(ANIMALS.items()):
    
    animal_dir = os.path.join(RAW_DIR, animal)
    
    # Skip if already downloaded enough
    if os.path.exists(animal_dir):
        existing = len([f for f in os.listdir(animal_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if existing >= TARGET_PER_ANIMAL:
            print(f"\n✅ {animal}: Already have {existing} images, skipping!")
            continue
    
    print(f"\n{'='*60}")
    print(f"  📥 [{animal_idx+1}/8] Downloading: {animal.upper()}")
    print(f"{'='*60}")
    
    os.makedirs(animal_dir, exist_ok=True)
    
    for q_idx, query in enumerate(queries):
        print(f"\n  🔍 Query {q_idx+1}/{len(queries)}: \"{query}\"")
        
        # Create temp dir for this query
        temp_dir = os.path.join(RAW_DIR, "_temp_download")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        try:
            crawler = BingImageCrawler(
                storage={'root_dir': temp_dir},
                feeder_threads=2,
                parser_threads=2,
                downloader_threads=4,
            )
            
            crawler.crawl(
                keyword=query,
                max_num=IMAGES_PER_QUERY,
                min_size=(100, 100),       # Skip tiny images
                file_idx_offset=q_idx * IMAGES_PER_QUERY
            )
            
            # Move downloaded images to animal folder
            moved = 0
            for fname in os.listdir(temp_dir):
                src = os.path.join(temp_dir, fname)
                # Rename to avoid conflicts
                new_name = f"{animal}_q{q_idx}_{fname}"
                dst = os.path.join(animal_dir, new_name)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
                    moved += 1
            
            print(f"     ✅ Got {moved} images")
            
        except Exception as e:
            print(f"     ⚠️ Error: {e}")
            continue
        
        finally:
            # Clean temp
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        # Check if we have enough
        current = len(os.listdir(animal_dir))
        if current >= TARGET_PER_ANIMAL + 200:  # Extra buffer
            print(f"     ✅ Have enough ({current} images), moving on!")
            break
    
    total = len([f for f in os.listdir(animal_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    print(f"\n  📊 {animal}: {total} raw images downloaded")

# ========================================
# PHASE 2: Validate images (remove broken/tiny ones)
# ========================================
print(f"\n{'='*60}")
print("  🔍 PHASE 2: Validating images")
print(f"{'='*60}")

for animal in ANIMALS:
    animal_dir = os.path.join(RAW_DIR, animal)
    if not os.path.exists(animal_dir):
        continue
    
    removed = 0
    valid = 0
    
    for fname in os.listdir(animal_dir):
        fpath = os.path.join(animal_dir, fname)
        
        try:
            with Image.open(fpath) as img:
                img.verify()
            
            # Re-open to actually load
            with Image.open(fpath) as img:
                img = img.convert("RGB")
                w, h = img.size
                
                # Remove if too small
                if w < 50 or h < 50:
                    os.remove(fpath)
                    removed += 1
                    continue
                
                # Remove if weird aspect ratio (probably not an animal photo)
                ratio = max(w, h) / min(w, h)
                if ratio > 5:
                    os.remove(fpath)
                    removed += 1
                    continue
                
                valid += 1
                
        except Exception:
            os.remove(fpath)
            removed += 1
    
    print(f"  {animal:12s}: {valid} valid, {removed} removed")

# ========================================
# PHASE 3: Remove duplicates
# ========================================
print(f"\n{'='*60}")
print("  🧹 PHASE 3: Removing duplicates")
print(f"{'='*60}")

for animal in ANIMALS:
    animal_dir = os.path.join(RAW_DIR, animal)
    if not os.path.exists(animal_dir):
        continue
    
    hashes = {}
    removed = 0
    
    for fname in sorted(os.listdir(animal_dir)):
        fpath = os.path.join(animal_dir, fname)
        if not os.path.isfile(fpath):
            continue
        
        # Hash the file
        hasher = hashlib.md5()
        with open(fpath, 'rb') as f:
            buf = f.read(65536)
            while buf:
                hasher.update(buf)
                buf = f.read(65536)
        file_hash = hasher.hexdigest()
        
        if file_hash in hashes:
            os.remove(fpath)
            removed += 1
        else:
            hashes[file_hash] = fpath
    
    remaining = len(os.listdir(animal_dir))
    print(f"  {animal:12s}: {removed} duplicates removed, {remaining} unique")

# ========================================
# PHASE 4: Resize all images to consistent size
# ========================================
print(f"\n{'='*60}")
print("  📐 PHASE 4: Resizing to 224x224")
print(f"{'='*60}")

for animal in ANIMALS:
    animal_dir = os.path.join(RAW_DIR, animal)
    if not os.path.exists(animal_dir):
        continue
    
    count = 0
    for fname in os.listdir(animal_dir):
        fpath = os.path.join(animal_dir, fname)
        try:
            with Image.open(fpath) as img:
                img = img.convert("RGB")
                img = img.resize((224, 224), Image.LANCZOS)
                
                # Save as jpg
                new_path = fpath.rsplit('.', 1)[0] + '.jpg'
                img.save(new_path, "JPEG", quality=92)
                
                # Remove original if different extension
                if new_path != fpath and os.path.exists(fpath):
                    os.remove(fpath)
                
                count += 1
        except Exception:
            if os.path.exists(fpath):
                os.remove(fpath)
    
    print(f"  {animal:12s}: {count} images resized ✅")

# ========================================
# PHASE 5: Limit to TARGET and split train/val
# ========================================
print(f"\n{'='*60}")
print(f"  📂 PHASE 5: Splitting into train/validation")
print(f"{'='*60}")

# Clean old dataset
if os.path.exists(FINAL_DIR):
    shutil.rmtree(FINAL_DIR)

total_train = 0
total_val = 0

for animal in ANIMALS:
    animal_dir = os.path.join(RAW_DIR, animal)
    if not os.path.exists(animal_dir):
        print(f"  ⚠️ {animal}: no images!")
        continue
    
    all_images = sorted([
        f for f in os.listdir(animal_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    # Limit to target
    if len(all_images) > TARGET_PER_ANIMAL:
        random.seed(42)
        random.shuffle(all_images)
        all_images = all_images[:TARGET_PER_ANIMAL]
    
    # Split
    random.seed(42)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * TRAIN_RATIO)
    
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]
    
    # Copy
    for split_name, img_list in [("train", train_imgs), ("validation", val_imgs)]:
        dst_dir = os.path.join(FINAL_DIR, split_name, animal)
        os.makedirs(dst_dir, exist_ok=True)
        for img_name in img_list:
            src = os.path.join(animal_dir, img_name)
            dst = os.path.join(dst_dir, img_name)
            shutil.copy2(src, dst)
    
    total_train += len(train_imgs)
    total_val += len(val_imgs)
    print(f"  {animal:12s}: {len(train_imgs)} train + {len(val_imgs)} val = {len(all_images)} total")

# ========================================
# FINAL SUMMARY
# ========================================
print(f"\n{'='*60}")
print(f"  📊 FRESH DATASET READY!")
print(f"{'='*60}")
print(f"  Total Training:   {total_train} images")
print(f"  Total Validation: {total_val} images")
print(f"  Grand Total:      {total_train + total_val} images")
print(f"  Animals:          {len(ANIMALS)}")
print(f"  All unique:       ✅ (deduplicated)")
print(f"  All validated:    ✅ (no broken images)")
print(f"  All resized:      ✅ (224×224 for quality)")
print(f"{'='*60}")
print(f"\n  ✅ Ready to train!")
print(f"  Run: py -3.10 distill.py")
print(f"\n  💡 You can delete 'raw_downloads/' to save space")