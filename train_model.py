import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, Sequence

# --- CONFIGURATION ---
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = 'data/train_images'
CSV_PATH = 'data/labels_train.csv'

# --- 1. RAM DATA GENERATOR ---
class AugmentedRamGenerator(Sequence):
    def __init__(self, X, y, batch_size, augment=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.indexes = np.arange(len(self.X))
        
        self.aug_pipeline = A.Compose([
            A.Rotate(limit=10, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.CLAHE(clip_limit=2.0, p=0.5),
            A.GaussNoise(p=0.2)
        ])

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X_batch = self.X[indexes]
        y_batch = self.y[indexes]
        
        if self.augment:
            X_aug = np.empty_like(X_batch)
            for i, img in enumerate(X_batch):
                img_uint8 = (img * 255).astype(np.uint8)
                augmented = self.aug_pipeline(image=img_uint8)['image']
                X_aug[i] = augmented.astype('float32') / 255.0
            return X_aug, y_batch
            
        return X_batch, y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

def load_data_into_ram(df, img_dir):
    print(f"Loading {len(df)} images into RAM...")
    images = []
    labels = []
    
    for i, row in df.iterrows():
        filename = str(row['image_id'])
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filename += '.jpeg'
            
        img_path = os.path.join(img_dir, filename)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(row['label'])
        except Exception:
            pass

    X = np.array(images, dtype='float32') / 255.0
    y = to_categorical(np.array(labels), num_classes=3)
    return X, y

if __name__ == "__main__":
    # 1. Load Data
    df = pd.read_csv(CSV_PATH)
    X, y = load_data_into_ram(df, DATA_DIR)
    
    # 2. Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Class Weights
    y_integers = np.argmax(y_train, axis=1)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    weights_dict = dict(enumerate(class_weights))
    print(f"Class Weights: {weights_dict}")

    # 4. Generators
    train_gen = AugmentedRamGenerator(X_train, y_train, BATCH_SIZE, augment=True)
    val_gen = AugmentedRamGenerator(X_val, y_val, BATCH_SIZE, augment=False)

    # 5. Build ResNet50 (Frozen)
    print("Building ResNet50...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False 
    
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(3, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 6. TRAIN (NO CALLBACKS = NO CRASHES)
    print("Starting Training (Safe Mode)...")
    # We removed 'callbacks=[...]' entirely
    model.fit(train_gen, 
              validation_data=val_gen, 
              epochs=EPOCHS, 
              class_weight=weights_dict)

    # 7. SAVE MANUALLY AT THE END
    print("Saving model...")
    model.save('best_model.keras')
    print("✅ DONE! Model saved as 'best_model.keras'")