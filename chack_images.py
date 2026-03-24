# check_images.py
# Shows a few sample images from each animal class

import matplotlib.pyplot as plt
import os
from PIL import Image

dataset_dir = os.path.join("dataset", "train")

if not os.path.exists(dataset_dir):
    print("❌ dataset/train/ not found! Run organize_data.py first.")
    exit()

animals = sorted(os.listdir(dataset_dir))
num_animals = len(animals)

fig, axes = plt.subplots(num_animals, 4, figsize=(12, 3 * num_animals))
fig.suptitle("Sample Images from Dataset", fontsize=16)

for row, animal in enumerate(animals):
    animal_path = os.path.join(dataset_dir, animal)
    images = os.listdir(animal_path)[:4]  # First 4 images
    
    for col in range(4):
        ax = axes[row][col] if num_animals > 1 else axes[col]
        
        if col < len(images):
            img_path = os.path.join(animal_path, images[col])
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                if col == 0:
                    ax.set_ylabel(animal.upper(), fontsize=14, fontweight='bold')
            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center')
        
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig("sample_images.png")
plt.show()
print("✅ Sample images saved as: sample_images.png")
print("Check if the images look correct!")