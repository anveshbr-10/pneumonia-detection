import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# ---------------- DATA ----------------
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    'data/train',
    target_size=(160, 160),
    batch_size=16,
    class_mode='categorical',
    classes=['NORMAL', 'BACTERIAL_PNEUMONIA', 'VIRAL_PNEUMONIA']
)

print("Class indices:", train_gen.class_indices)

# ---------------- MODEL ----------------
base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(160, 160, 3)
)

# 🔒 Freeze everything first
base_model.trainable = False

# 🔓 Unfreeze ONLY last DenseNet block
for layer in base_model.layers:
    if 'conv5' in layer.name:
        layer.trainable = True

x = GlobalAveragePooling2D()(base_model.output)
output = Dense(3, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------- CLASS WEIGHTS ----------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1, 2]),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# ---------------- TRAIN ----------------
model.fit(
    train_gen,
    epochs=2,
    class_weight=class_weights
)

model.save("pneumonia_model_balanced.h5")
print("✅ Balanced model saved")
