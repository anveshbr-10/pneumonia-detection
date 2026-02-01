import tensorflow as tf

# Load model
model = tf.keras.models.load_model('pneumonia_model_fast.h5')

# Test data generator (NO augmentation)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    'data/test',
    target_size=(160, 160),
    batch_size=16,
    class_mode='categorical',
    shuffle=False,
    classes=['NORMAL', 'BACTERIAL_PNEUMONIA', 'VIRAL_PNEUMONIA']
)

print("Class indices:", test_gen.class_indices)

# Predict on all test images
predictions = model.predict(test_gen)

# Decode predictions
import numpy as np
class_names = ['NORMAL', 'BACTERIAL_PNEUMONIA', 'VIRAL_PNEUMONIA']
predicted_classes = np.argmax(predictions, axis=1)

# Show sample results
for i in range(10):  # print first 10 predictions
    print(
        test_gen.filenames[i],
        "→",
        class_names[predicted_classes[i]]
    )
