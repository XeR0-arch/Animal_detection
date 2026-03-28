# run_all.py
# Runs the entire pipeline from scratch
# Usage: py -3.10 run_all.py

import subprocess
import sys
import os

py = sys.executable

steps = [
    ("📥 Step 1: Download fresh images",    "download_fresh.py"),
    ("🧹 Step 2: Fix duplicates",           "fix_duplicates.py"),
    ("🧠 Step 3: Train & Distill",          "distill.py"),
    ("🔧 Step 4: Convert for ESP32",        "convert.py"),
    ("📊 Step 5: Generate report PDF",      "generate_report.py"),
]

print("=" * 55)
print("  🚀 WILDEDGE AI — FULL PIPELINE")
print("=" * 55)

for i, (desc, script) in enumerate(steps):
    print(f"\n{'='*55}")
    print(f"  {desc}")
    print(f"{'='*55}\n")
    
    if not os.path.exists(script):
        print(f"  ⚠️ {script} not found, skipping...")
        continue
    
    result = subprocess.run([py, script])
    
    if result.returncode != 0:
        print(f"\n  ❌ {script} failed! Fix the error and run again.")
        print(f"     Or run individually: py -3.10 {script}")
        break
    
    print(f"\n  ✅ {desc} — DONE!")

print(f"\n{'='*55}")
print(f"  🎉 PIPELINE COMPLETE!")
print(f"{'='*55}")