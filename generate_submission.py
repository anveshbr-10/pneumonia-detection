import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
IMG_SIZE = 128
# We point to 'data' and let the script hunt for images inside 'test_images'
BASE_DIR = os.path.join(os.getcwd(), 'data', 'test_images') 
MODEL_PATH = 'best_model.keras'
OUTPUT_CSV = 'submission.csv'

# --- UTILS ---
def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

if __name__ == "__main__":
    print(f"🔍 SEARCHING FOR IMAGES IN: {BASE_DIR}")
    
    # 1. FIND IMAGES (RECURSIVE SEARCH)
    test_files = []
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Save full path
                full_path = os.path.join(root, file)
                test_files.append(full_path)

    # 2. CHECK IF EMPTY
    if len(test_files) == 0:
        print("\n❌ CRITICAL ERROR: No images found!")
        print(f"   I looked inside: {BASE_DIR}")
        print("   Please check:")
        print("   1. Did you paste the 'test_images' folder inside 'data'?")
        print("   2. does 'data/test_images' actually contain .jpg/.png files?")
        exit()
    
    print(f"✅ FOUND {len(test_files)} IMAGES. Loading Model...")

    # 3. LOAD MODEL
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model '{MODEL_PATH}' not found.")
        exit()
    model = load_model(MODEL_PATH)

    results = []
    print("🚀 Starting Predictions...")

    # 4. PREDICT LOOP
    for i, filepath in enumerate(test_files):
        # Extract filename for CSV (e.g., 'patient123.jpg')
        filename = os.path.basename(filepath)
        
        img = load_and_preprocess_image(filepath)
        if img is None:
            continue
            
        img_batch = np.expand_dims(img, axis=0).astype('float32')
        preds = model.predict(img_batch, verbose=0)
        
        predicted_class = np.argmax(preds)
        
        # LABEL MAP (Adjust if your hackathon needs strings vs numbers)
        # 0: Normal, 1: Bacterial, 2: Viral
        label_map = {0: "Normal", 1: "Bacterial Pneumonia", 2: "Viral Pneumonia"}
        
        results.append({
            "image_id": filename,
            "label": predicted_class, # Submitting the number (Safe bet)
            "label_text": label_map[predicted_class] # Extra column just for you to check
        })
        
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(test_files)}...")

    # 5. SAVE
    if len(results) > 0:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print("\n" + "="*40)
        print(f"🎉 DONE! Saved {len(results)} rows to '{OUTPUT_CSV}'")
        print("="*40)
        print(df.head())
    else:
        print("❌ Prediction loop finished but results are empty. Failed to process images.")