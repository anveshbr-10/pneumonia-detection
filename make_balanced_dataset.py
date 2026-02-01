import os
import shutil
import random

SOURCE_DIR = "data/train"
TARGET_DIR = "data/train_balanced"

CLASSES = ["NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA"]

# Create target folders
for cls in CLASSES:
    os.makedirs(os.path.join(TARGET_DIR, cls), exist_ok=True)

# Find minimum class count
counts = {}
for cls in CLASSES:
    counts[cls] = len(os.listdir(os.path.join(SOURCE_DIR, cls)))

min_count = min(counts.values())
print("Using balanced count:", min_count)

# Copy equal number of images
for cls in CLASSES:
    src = os.path.join(SOURCE_DIR, cls)
    dst = os.path.join(TARGET_DIR, cls)

    images = os.listdir(src)
    random.shuffle(images)
    selected = images[:min_count]

    for img in selected:
        shutil.copy(
            os.path.join(src, img),
            os.path.join(dst, img)
        )

print("✅ Balanced dataset created")
