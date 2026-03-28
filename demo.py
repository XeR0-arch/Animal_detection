import tensorflow as tf
import numpy as np
import os
from PIL import Image

model = tf.keras.models.load_model("student_model.keras")
with open("class_names.txt") as f:
    CLASS_NAMES = [l.strip() for l in f.readlines()]

val_dir = os.path.join("dataset", "validation")
dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir, image_size=(96,96), batch_size=1,
    label_mode='categorical', shuffle=True, seed=42
)

print("\n" + "="*50)
print("  🌿 WildEdge AI — LIVE DEMO")
print("  ESP32-CAM Edge Inference Simulator")
print("="*50)

correct = 0
total = 0

for image, label in dataset.take(10):
    pred = model.predict(image/255.0, verbose=0)
    pred_class = CLASS_NAMES[np.argmax(pred)]
    actual = CLASS_NAMES[np.argmax(label[0])]
    conf = np.max(pred)*100
    correct += pred_class == actual
    total += 1
    
    status = "✅ CORRECT" if pred_class == actual else "❌ WRONG"
    print(f"\n  📸 Image captured (96×96 pixels)")
    print(f"  🧠 Running inference on Student Model (274 KB)...")
    print(f"  🎯 Detected: {pred_class.upper()} ({conf:.1f}% confidence)")
    print(f"  📋 Actual:   {actual.upper()}")
    print(f"  {status}")
    print(f"  ─────────────────────────────")
    
    import time
    time.sleep(1.5)  # Dramatic pause for demo

print(f"\n  FINAL: {correct}/{total} correct ({correct/total*100:.0f}%)")
print(f"  Model: 274KB | No cloud | Runs on $6 ESP32-CAM")
print("="*50)