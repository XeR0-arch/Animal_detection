# train.py — TRANSFER LEARNING VERSION
# Uses MobileNetV2 (Google's pretrained model)
# Knows what animals look like BEFORE we even start!

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import matplotlib.pyplot as plt

# ========================================
# CHECK GPU
# ========================================
print("=" * 55)
print("  🐾 JUNGLE ANIMAL DETECTOR — Transfer Learning")
print("=" * 55)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n🎮 GPU DETECTED: {gpus[0].name}")
    print("   Training will be FAST! ⚡")
    # Prevent GPU memory explosion
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("\n💻 No GPU detected — using CPU")
    print("   Training will be slower but still works!")

# ========================================
# CONFIG
# ========================================
DATASET_DIR = "dataset"
IMG_SIZE = 96              # MobileNetV2 works great at 96x96
BATCH_SIZE = 16
EPOCHS = 40

# ========================================
# STEP 1: Load Images
# ========================================
print("\n📂 Loading images...")

train_data = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
    seed=42
)

val_data = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "validation"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    seed=42
)

CLASS_NAMES = train_data.class_names
NUM_CLASSES = len(CLASS_NAMES)
print(f"✅ Found {NUM_CLASSES} classes: {CLASS_NAMES}")

# Count total images
train_count = sum(1 for _ in train_data.unbatch())
val_count = sum(1 for _ in val_data.unbatch())
print(f"   Training:   {train_count} images")
print(f"   Validation: {val_count} images")

# ========================================
# STEP 2: Preprocess for MobileNetV2
# MobileNetV2 expects pixels in [-1, 1] range
# ========================================
print("\n🔧 Preprocessing...")

def preprocess(image, label):
    # MobileNetV2 preprocess: scales pixels from [0,255] to [-1,1]
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

train_data = train_data.map(preprocess).cache().prefetch(tf.data.AUTOTUNE)
val_data = val_data.map(preprocess).cache().prefetch(tf.data.AUTOTUNE)

# ========================================
# STEP 3: Data Augmentation
# ========================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom((-0.15, 0.15)),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1),
])

# ========================================
# STEP 4: Build Model with TRANSFER LEARNING
# ========================================
print("\n🏗️  Building Transfer Learning model...")
print("   Downloading MobileNetV2 (first time only)...\n")

# Load MobileNetV2 — trained on 1.4 MILLION images by Google
# alpha=0.35 makes it SMALL (for ESP32-CAM!)
# include_top=False removes Google's classifier (we add our own)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=0.35,                    # Width multiplier — smaller = fits ESP32
    include_top=False,             # Remove the 1000-class head
    weights='imagenet'             # Use pretrained weights!
)

# FREEZE the base model — don't change what it already learned
base_model.trainable = False

print(f"   ✅ MobileNetV2 loaded!")
print(f"   Base model layers: {len(base_model.layers)}")
print(f"   Base model params: {base_model.count_params():,} (frozen)")

# Build full model
model = models.Sequential([
    # Input
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    # Augmentation (only during training)
    data_augmentation,
    
    # MobileNetV2 base (frozen — already knows animals!)
    base_model,
    
    # Our custom classifier on top
    layers.GlobalAveragePooling2D(),     # Smarter than Flatten
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

# ========================================
# STEP 5: Train — Phase 1 (train only the top)
# ========================================
print(f"\n🎯 Phase 1: Training classifier head ({EPOCHS} epochs)...")
print("   (Base model is FROZEN — only learning our animal classes)\n")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop],
    verbose=1
)

phase1_val_acc = max(history1.history['val_accuracy'])
print(f"\n   Phase 1 Best Validation Accuracy: {phase1_val_acc*100:.1f}%")

# ========================================
# STEP 6: Fine-tune — Phase 2 (unlock some base layers)
# ========================================
print(f"\n🔓 Phase 2: Fine-tuning (unfreezing top layers of MobileNetV2)...")
print("   Teaching the base model to focus on JUNGLE animals specifically\n")

# Unfreeze the LAST 20 layers of MobileNetV2
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

trainable_count = sum(1 for layer in model.layers if layer.trainable)
print(f"   Unfroze top 20 layers for fine-tuning")

# Use VERY small learning rate — don't destroy what it already knows!
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # 10x smaller!
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,             # Fewer epochs for fine-tuning
    callbacks=[early_stop],
    verbose=1
)

# ========================================
# STEP 7: Final Results
# ========================================
val_loss, val_acc = model.evaluate(val_data, verbose=0)

# Combine histories for plotting
full_history = {}
for key in history1.history:
    full_history[key] = history1.history[key] + history2.history[key]

best_val_acc = max(full_history['val_accuracy'])

print(f"\n{'=' * 55}")
print(f"  📊 FINAL RESULTS")
print(f"{'=' * 55}")
print(f"  Best Validation Accuracy: {best_val_acc*100:.1f}%")
print(f"  Final Validation Accuracy: {val_acc*100:.1f}%")
print(f"  Classes: {CLASS_NAMES}")
print(f"{'=' * 55}")

if val_acc > 0.80:
    print("  ✅ すごい！(Sugoi!) Excellent! 🎉")
elif val_acc > 0.65:
    print("  ✅ いいね！(Ii ne!) Good! Will work on ESP32.")
elif val_acc > 0.50:
    print("  ⚠️  まあまあ (Maa maa) — Decent for a start.")
else:
    print("  😐 Need more data or different animals.")

# ========================================
# STEP 8: Save model
# ========================================
model.save("animal_model.keras")
model_size = os.path.getsize("animal_model.keras") / 1024
print(f"\n💾 Model saved: animal_model.keras ({model_size:.0f} KB)")

with open("class_names.txt", "w") as f:
    for name in CLASS_NAMES:
        f.write(name + "\n")
print(f"📝 Classes: {CLASS_NAMES}")

# ========================================
# STEP 9: Plot
# ========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(len(full_history['accuracy']))

# Mark where fine-tuning started
phase1_epochs = len(history1.history['accuracy'])

ax1.plot(epochs_range, full_history['accuracy'], 'b-', label='Train', linewidth=2)
ax1.plot(epochs_range, full_history['val_accuracy'], 'r-', label='Validation', linewidth=2)
ax1.axvline(x=phase1_epochs, color='green', linestyle='--', label='Fine-tune start')
ax1.set_title('Accuracy', fontsize=14)
ax1.set_xlabel('Epoch')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, full_history['loss'], 'b-', label='Train', linewidth=2)
ax2.plot(epochs_range, full_history['val_loss'], 'r-', label='Validation', linewidth=2)
ax2.axvline(x=phase1_epochs, color='green', linestyle='--', label='Fine-tune start')
ax2.set_title('Loss', fontsize=14)
ax2.set_xlabel('Epoch')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle(f'Transfer Learning — Val Accuracy: {val_acc*100:.1f}%',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("training_results.png", dpi=100)
plt.show()

# ========================================
# STEP 10: Test Predictions (THE REAL TEST!)
# ========================================
print("\n🔍 Testing on validation images...\n")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "validation"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    label_mode='categorical',
    shuffle=True,
    seed=99
)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

correct = 0
total = 0

for i, (image, label) in enumerate(test_dataset.take(12)):
    # Preprocess same way as training
    img_processed = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    
    prediction = model.predict(img_processed, verbose=0)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    actual_class = CLASS_NAMES[np.argmax(label[0])]
    confidence = np.max(prediction) * 100

    if predicted_class == actual_class:
        correct += 1
    total += 1

    # Show original image (not preprocessed)
    axes[i].imshow(image[0].numpy().astype("uint8"))
    color = "green" if predicted_class == actual_class else "red"
    axes[i].set_title(
        f"Pred: {predicted_class} ({confidence:.0f}%)\nActual: {actual_class}",
        color=color, fontsize=10, fontweight='bold'
    )
    axes[i].axis('off')

plt.suptitle(
    f"Predictions: {correct}/{total} correct ({correct/total*100:.0f}%)\n"
    f"Green = Right, Red = Wrong",
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig("test_predictions.png", dpi=100)
plt.show()

print(f"\n📸 Real test: {correct}/{total} correct ({correct/total*100:.0f}%)")

print(f"\n{'=' * 55}")
print(f"  🎉 TRAINING COMPLETE!")
print(f"{'=' * 55}")
print(f"  Approach:  Transfer Learning (MobileNetV2)")
print(f"  Model:     animal_model.keras ({model_size:.0f} KB)")
print(f"  Classes:   {CLASS_NAMES}")
print(f"  Accuracy:  {val_acc*100:.1f}%")
print(f"  Real test: {correct}/{total} correct")
print(f"{'=' * 55}")
print(f"\n  次のステップ: py -3.10 convert.py")