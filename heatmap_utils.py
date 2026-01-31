import numpy as np
import cv2
import tensorflow as tf

def get_heatmap(model, img_array, class_index):
    """
    Manual 'Split & Watch' Grad-CAM with Smoothing.
    """
    # 1. SPLIT THE MODEL (Base vs Head)
    base_model = model.layers[1] # ResNet50
    classifier_layers = model.layers[2:] # Head

    # 2. WATCH THE GRADIENTS
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        
        # Run Base
        conv_outputs = base_model(inputs, training=False)
        tape.watch(conv_outputs)
        
        # Run Head
        x = conv_outputs
        for layer in classifier_layers:
            x = layer(x, training=False)
        
        predictions = x
        loss = predictions[:, class_index]

    # 3. CALCULATE GRADIENTS
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 4. GENERATE HEATMAP
    conv_outputs = conv_outputs[0]
    
    # Weight the channels
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU (Keep only positive influence)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()

    return heatmap

def overlay_heatmap(original_img, heatmap, alpha=0.5):
    """
    Overlays ONLY the hot spots (Red/Yellow) and makes the background transparent.
    """
    # 1. Ensure Original Image is uint8 (0-255)
    if original_img.dtype != np.uint8:
        original_img = np.uint8(255 * original_img)

    # 2. Resize Heatmap to match Original Image Size
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # 3. CLEANUP: Thresholding (The "Pro" Fix)
    # We hide any heat that is less than 35% intensity. This removes the "Blue Blob".
    heatmap = np.where(heatmap < 0.35, 0, heatmap)

    # 4. Apply ColorMap (JET is standard: Blue=Cold, Red=Hot)
    # Scale to 0-255
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Convert BGR to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # 5. SMART BLEND: Only blend where the heatmap exists
    # Create a mask where heatmap > 0
    mask = heatmap > 0
    
    # Create a copy of the original image
    superimposed_img = original_img.copy()
    
    # Apply the blend ONLY to the masked areas
    # formula: Output = alpha * Heatmap + (1 - alpha) * Original
    superimposed_img[mask] = cv2.addWeighted(
        heatmap_colored[mask], alpha, 
        original_img[mask], 1 - alpha, 
        0
    ).squeeze() # Squeeze ensures dimensions match

    return superimposed_img