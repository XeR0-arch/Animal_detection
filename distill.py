# distill.py
# Knowledge Distillation: Big Teacher → Small Student
# Result: Tiny model with big-model accuracy

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import matplotlib.pyplot as plt

# ========================================
# GPU CHECK
# ========================================
print("=" * 60)
print("  🧠 KNOWLEDGE DISTILLATION PIPELINE")
print("  Big Teacher → Smart Small Student → ESP32-CAM")
print("=" * 60)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n🎮 GPU: {gpus[0].name}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("\n💻 CPU mode — will be slower but works!")

# ========================================
# CONFIG
# ========================================
DATASET_DIR = "dataset"
TEACHER_IMG_SIZE = 160     # Teacher sees BIGGER images
STUDENT_IMG_SIZE = 96      # Student sees what ESP32 will see
BATCH_SIZE = 16
TEACHER_EPOCHS = 50
STUDENT_EPOCHS = 60
DISTILL_EPOCHS = 80        # Distillation needs more epochs

# Distillation hyperparameters
TEMPERATURE = 5.0          # Higher = softer probability distributions
ALPHA = 0.7                # How much to trust teacher vs real labels
                           # 0.7 = 70% teacher knowledge, 30% real labels

# ========================================
# STEP 1: Load Data (two versions — big for teacher, small for student)
# ========================================
print("\n📂 Loading dataset...")

# Teacher dataset (bigger images)
teacher_train = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    image_size=(TEACHER_IMG_SIZE, TEACHER_IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
    seed=42
)

teacher_val = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "validation"),
    image_size=(TEACHER_IMG_SIZE, TEACHER_IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    seed=42
)

# Student dataset (smaller images — what ESP32 camera actually sees)
student_train = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    image_size=(STUDENT_IMG_SIZE, STUDENT_IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
    seed=42
)

student_val = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "validation"),
    image_size=(STUDENT_IMG_SIZE, STUDENT_IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    seed=42
)

CLASS_NAMES = teacher_train.class_names
NUM_CLASSES = len(CLASS_NAMES)
print(f"✅ {NUM_CLASSES} classes: {CLASS_NAMES}")

# Preprocess
def preprocess_mobilenet(image, label):
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

def preprocess_simple(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

teacher_train_p = teacher_train.map(preprocess_mobilenet).cache().prefetch(tf.data.AUTOTUNE)
teacher_val_p = teacher_val.map(preprocess_mobilenet).cache().prefetch(tf.data.AUTOTUNE)
student_train_p = student_train.map(preprocess_simple).cache().prefetch(tf.data.AUTOTUNE)
student_val_p = student_val.map(preprocess_simple).cache().prefetch(tf.data.AUTOTUNE)

# ========================================
# PHASE 1: TRAIN THE BIG TEACHER
# ========================================
print(f"\n{'='*60}")
print("  📚 PHASE 1: Training BIG Teacher Model")
print(f"  Image size: {TEACHER_IMG_SIZE}x{TEACHER_IMG_SIZE}")
print(f"{'='*60}\n")

# Data augmentation for teacher
teacher_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom((-0.2, 0.2)),
    layers.RandomTranslation(0.15, 0.15),
    layers.RandomContrast(0.2),
])

# Big MobileNetV2 (alpha=1.0 — full size!)
teacher_base = tf.keras.applications.MobileNetV2(
    input_shape=(TEACHER_IMG_SIZE, TEACHER_IMG_SIZE, 3),
    alpha=1.0,                 # FULL size — don't care, it won't go on ESP32
    include_top=False,
    weights='imagenet'
)
teacher_base.trainable = False

teacher_model = models.Sequential([
    layers.Input(shape=(TEACHER_IMG_SIZE, TEACHER_IMG_SIZE, 3)),
    teacher_augmentation,
    teacher_base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

print("🎯 Phase 1a: Training teacher head (frozen base)...")

teacher_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=10, restore_best_weights=True
)

h1 = teacher_model.fit(
    teacher_train_p, validation_data=teacher_val_p,
    epochs=TEACHER_EPOCHS, callbacks=[early_stop], verbose=1
)

phase1a_acc = max(h1.history['val_accuracy'])
print(f"\n   Phase 1a accuracy: {phase1a_acc*100:.1f}%")

# Fine-tune teacher
print("\n🔓 Phase 1b: Fine-tuning teacher (unfreezing layers)...")

teacher_base.trainable = True
for layer in teacher_base.layers[:-30]:
    layer.trainable = False

teacher_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.00005),  # Very small LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

h2 = teacher_model.fit(
    teacher_train_p, validation_data=teacher_val_p,
    epochs=25, callbacks=[early_stop], verbose=1
)

teacher_val_loss, teacher_val_acc = teacher_model.evaluate(teacher_val_p, verbose=0)
print(f"\n   ✅ TEACHER Final Accuracy: {teacher_val_acc*100:.1f}%")

teacher_model.save("teacher_model.keras")
teacher_size = os.path.getsize("teacher_model.keras") / 1024
print(f"   💾 Teacher saved: {teacher_size:.0f} KB (too big for ESP32, that's OK!)")

# ========================================
# PHASE 2: BUILD THE TINY STUDENT
# ========================================
print(f"\n{'='*60}")
print("  🎒 PHASE 2: Building TINY Student Model")
print(f"  Image size: {STUDENT_IMG_SIZE}x{STUDENT_IMG_SIZE}")
print(f"  Target size: < 200KB after quantization")
print(f"{'='*60}\n")

student_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom((-0.1, 0.1)),
    layers.RandomContrast(0.1),
])

# Tiny but efficient architecture
# Using depthwise separable convolutions (like MobileNet but hand-crafted tiny)
def build_student():
    inputs = layers.Input(shape=(STUDENT_IMG_SIZE, STUDENT_IMG_SIZE, 3))
    
    x = student_augmentation(inputs)
    
    # Block 1: Regular conv
    x = layers.Conv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # → 48x48x16
    
    # Block 2: Depthwise separable (efficient!)
    x = layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(24, (1, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # → 24x24x24
    
    # Block 3: Depthwise separable
    x = layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, (1, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # → 12x12x32
    
    # Block 4: Depthwise separable
    x = layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(48, (1, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # → 6x6x48
    
    # Block 5: Depthwise separable
    x = layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (1, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # → 6x6x64
    
    # Classifier
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(NUM_CLASSES)(x)  # NO softmax! (needed for distillation)
    
    return models.Model(inputs, outputs, name="student")

student_model = build_student()
student_model.summary()

student_params = student_model.count_params()
print(f"\n📊 Student parameters: {student_params:,}")
print(f"   Estimated after quantization: ~{student_params / 1024:.0f} KB")

# ========================================
# PHASE 3: KNOWLEDGE DISTILLATION
# ========================================
print(f"\n{'='*60}")
print("  🧪 PHASE 3: Knowledge Distillation")
print(f"  Temperature: {TEMPERATURE}")
print(f"  Alpha: {ALPHA} (teacher weight)")
print(f"  Teacher accuracy: {teacher_val_acc*100:.1f}%")
print(f"  Goal: Transfer that accuracy to tiny student!")
print(f"{'='*60}\n")

# Custom distillation loss
class DistillationLoss(tf.keras.losses.Loss):
    def __init__(self, temperature, alpha):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = tf.keras.losses.KLDivergence()
        self.ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    def call(self, y_true_and_teacher, y_pred_student_logits):
        # y_true_and_teacher is a concatenation:
        # first NUM_CLASSES = true labels
        # next NUM_CLASSES = teacher soft labels
        y_true = y_true_and_teacher[:, :NUM_CLASSES]
        teacher_logits = y_true_and_teacher[:, NUM_CLASSES:]
        
        # Soft targets from teacher
        teacher_soft = tf.nn.softmax(teacher_logits / self.temperature)
        student_soft = tf.nn.softmax(y_pred_student_logits / self.temperature)
        
        # Distillation loss (learn from teacher)
        distill_loss = self.kl_loss(teacher_soft, student_soft) * (self.temperature ** 2)
        
        # Hard label loss (learn from ground truth)
        hard_loss = self.ce_loss(y_true, y_pred_student_logits)
        
        # Combine
        return self.alpha * distill_loss + (1 - self.alpha) * hard_loss

# Generate teacher predictions (soft labels) for ALL training data
print("📚 Generating teacher's soft labels for all training data...")

# We need to get teacher predictions at student resolution too
# Load training data at teacher resolution for predictions
teacher_train_for_pred = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    image_size=(TEACHER_IMG_SIZE, TEACHER_IMG_SIZE),
    batch_size=1,
    label_mode='categorical',
    shuffle=False,        # IMPORTANT: same order as student data!
    seed=42
)

# Also load student data in same order
student_train_ordered = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    image_size=(STUDENT_IMG_SIZE, STUDENT_IMG_SIZE),
    batch_size=1,
    label_mode='categorical',
    shuffle=False,
    seed=42
)

# Get teacher predictions (logits before softmax)
# First, create a teacher model that outputs logits
teacher_logit_model = models.Model(
    inputs=teacher_model.input,
    outputs=teacher_model.layers[-1].output  # Before softmax
)

print("   Computing teacher predictions...")
teacher_predictions = []
true_labels = []

for images, labels in teacher_train_for_pred.map(preprocess_mobilenet):
    pred = teacher_model.predict(images, verbose=0)
    # Convert to logits (inverse of softmax, approximately)
    pred_clipped = np.clip(pred, 1e-7, 1.0)
    logits = np.log(pred_clipped)
    teacher_predictions.append(logits[0])
    true_labels.append(labels[0].numpy())

teacher_predictions = np.array(teacher_predictions)
true_labels = np.array(true_labels)
print(f"   ✅ Got {len(teacher_predictions)} teacher predictions")

# Get student training images
student_images = []
for images, _ in student_train_ordered.map(preprocess_simple):
    student_images.append(images[0].numpy())
student_images = np.array(student_images)
print(f"   ✅ Got {len(student_images)} student images")

# Combine true labels and teacher predictions
combined_labels = np.concatenate([true_labels, teacher_predictions], axis=1)
print(f"   Combined label shape: {combined_labels.shape}")
print(f"   (first {NUM_CLASSES} = true labels, next {NUM_CLASSES} = teacher logits)")

# Create distillation dataset
distill_dataset = tf.data.Dataset.from_tensor_slices((student_images, combined_labels))
distill_dataset = distill_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Compile student with distillation loss
print("\n🎯 Training student with teacher's knowledge...\n")

distill_loss = DistillationLoss(temperature=TEMPERATURE, alpha=ALPHA)

student_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=distill_loss,
)

# Learning rate scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Train with distillation
distill_history = student_model.fit(
    distill_dataset,
    epochs=DISTILL_EPOCHS,
    callbacks=[lr_scheduler],
    verbose=1
)

# ========================================
# PHASE 4: FINE-TUNE STUDENT ON REAL LABELS
# ========================================
print(f"\n{'='*60}")
print("  🎯 PHASE 4: Fine-tuning Student on Real Labels")
print(f"{'='*60}\n")

# Now train normally to sharpen predictions
# Create a NEW model with softmax for normal training
student_final = models.Model(
    inputs=student_model.input,
    outputs=layers.Activation('softmax')(student_model.output)
)

student_final.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop2 = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=12, restore_best_weights=True
)

lr_scheduler2 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.5, patience=5, min_lr=0.00001, verbose=1
)

finetune_history = student_final.fit(
    student_train_p,
    validation_data=student_val_p,
    epochs=40,
    callbacks=[early_stop2, lr_scheduler2],
    verbose=1
)

# ========================================
# PHASE 5: EVALUATE EVERYTHING
# ========================================
print(f"\n{'='*60}")
print("  📊 PHASE 5: Final Comparison")
print(f"{'='*60}\n")

# Teacher accuracy
teacher_loss, teacher_acc = teacher_model.evaluate(teacher_val_p, verbose=0)

# Student accuracy
student_loss, student_acc = student_final.evaluate(student_val_p, verbose=0)

print(f"  ┌──────────────────────────────────────────┐")
print(f"  │  MODEL          ACCURACY    SIZE          │")
print(f"  ├──────────────────────────────────────────┤")
print(f"  │  Teacher        {teacher_acc*100:5.1f}%      ~{teacher_size:.0f} KB     │")
print(f"  │  Student        {student_acc*100:5.1f}%      ~{student_params/1024:.0f} KB      │")
print(f"  │  Student (q8)   ~{student_acc*100-2:.0f}%       ~{student_params/1024/4:.0f} KB      │")
print(f"  │  ESP32 limit     —          500 KB        │")
print(f"  └──────────────────────────────────────────┘")

# Save student
student_final.save("student_model.keras")
student_size = os.path.getsize("student_model.keras") / 1024
print(f"\n💾 Student saved: student_model.keras ({student_size:.0f} KB)")

with open("class_names.txt", "w") as f:
    for name in CLASS_NAMES:
        f.write(name + "\n")

# ========================================
# PHASE 6: Visual Test (12 images)
# ========================================
print("\n🔍 Testing both models on same images...\n")

test_dataset_teacher = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "validation"),
    image_size=(TEACHER_IMG_SIZE, TEACHER_IMG_SIZE),
    batch_size=1, label_mode='categorical', shuffle=True, seed=77
)

test_dataset_student = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "validation"),
    image_size=(STUDENT_IMG_SIZE, STUDENT_IMG_SIZE),
    batch_size=1, label_mode='categorical', shuffle=True, seed=77
)

fig, axes = plt.subplots(3, 4, figsize=(18, 14))
axes = axes.flatten()

teacher_correct = 0
student_correct = 0
total = 0

for i, ((t_img, t_lbl), (s_img, s_lbl)) in enumerate(
    zip(test_dataset_teacher.take(12), test_dataset_student.take(12))
):
    actual = CLASS_NAMES[np.argmax(t_lbl[0])]
    
    # Teacher prediction
    t_pred = teacher_model.predict(
        tf.keras.applications.mobilenet_v2.preprocess_input(t_img), verbose=0
    )
    t_class = CLASS_NAMES[np.argmax(t_pred)]
    t_conf = np.max(t_pred) * 100
    
    # Student prediction
    s_pred = student_final.predict(s_img / 255.0, verbose=0)
    s_class = CLASS_NAMES[np.argmax(s_pred)]
    s_conf = np.max(s_pred) * 100
    
    if t_class == actual: teacher_correct += 1
    if s_class == actual: student_correct += 1
    total += 1
    
    # Show student image (what ESP32 sees)
    axes[i].imshow(s_img[0].numpy().astype("uint8"))
    
    s_color = "green" if s_class == actual else "red"
    t_mark = "✓" if t_class == actual else "✗"
    
    axes[i].set_title(
        f"Actual: {actual}\n"
        f"Student: {s_class} ({s_conf:.0f}%) {'✓' if s_class == actual else '✗'}\n"
        f"Teacher: {t_class} ({t_conf:.0f}%) {t_mark}",
        color=s_color, fontsize=9, fontweight='bold'
    )
    axes[i].axis('off')

plt.suptitle(
    f"Teacher: {teacher_correct}/{total} | Student: {student_correct}/{total}\n"
    f"Student learned {student_correct/max(teacher_correct,1)*100:.0f}% of teacher's ability!",
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig("distillation_results.png", dpi=100)
plt.show()

# ========================================
# FINAL SUMMARY
# ========================================
knowledge_transfer = student_acc / max(teacher_acc, 0.01) * 100

print(f"\n{'='*60}")
print(f"  🎉 DISTILLATION COMPLETE!")
print(f"{'='*60}")
print(f"  Teacher:  {teacher_acc*100:.1f}% accuracy ({teacher_size:.0f} KB)")
print(f"  Student:  {student_acc*100:.1f}% accuracy ({student_size:.0f} KB)")
print(f"  Transfer: {knowledge_transfer:.0f}% of teacher's knowledge retained")
print(f"  Size reduction: {teacher_size/student_size:.1f}x smaller!")
print(f"")
print(f"  Classes: {CLASS_NAMES}")
print(f"  Real test: Teacher {teacher_correct}/12, Student {student_correct}/12")
print(f"{'='*60}")

if student_acc > 0.75:
    print(f"\n  ✅ 完璧！(Kanpeki! — Perfect!)")
    print(f"     Student is ready for ESP32!")
elif student_acc > 0.65:
    print(f"\n  ✅ いいね！(Ii ne! — Nice!)")
    print(f"     Good enough for ESP32 deployment!")
elif student_acc > 0.50:
    print(f"\n  ⚠️  まあまあ — Can improve with more data")
else:
    print(f"\n  😐 Need more training data")

print(f"\n  次のステップ: py -3.10 convert.py")
print(f"  (Convert student model to ESP32 format)")