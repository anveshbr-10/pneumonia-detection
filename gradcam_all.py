"""
FIXED Grad-CAM for Pneumonia Detection - Batch Processing
Properly detects conv layers and generates accurate heatmaps
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from pathlib import Path

class PneumoniaGradCAM:
    """
    FIXED Grad-CAM implementation for batch processing pneumonia X-ray images
    """
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM for pneumonia detection
        
        Args:
            model: Trained Keras model
            layer_name: Target convolutional layer (auto-detects if None)
        """
        self.model = model
        self.class_names = ['NORMAL', 'BACTERIAL_PNEUMONIA', 'VIRAL_PNEUMONIA']
        
        # Auto-detect last conv layer if not specified - FIXED
        if layer_name is None:
            print("\n🔍 Detecting convolutional layers...")
            conv_layers = []
            for layer in model.layers:
                # Check if layer has 4D output (batch, height, width, channels)
                if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                    conv_layers.append(layer.name)
                    print(f"  Found: {layer.name} - {layer.output_shape}")
            
            if not conv_layers:
                raise ValueError(
                    "❌ No convolutional layers found in model! "
                    "Model might not support Grad-CAM."
                )
            
            # Use the last conv layer
            layer_name = conv_layers[-1]
            print(f"\n✓ Selected layer: {layer_name}")
        
        self.layer_name = layer_name
        
        # Create gradient model - FIXED error handling
        try:
            target_layer = model.get_layer(layer_name)
            self.grad_model = keras.models.Model(
                inputs=[model.input],
                outputs=[target_layer.output, model.output]
            )
            print(f"✓ Grad-CAM model created successfully")
        except Exception as e:
            raise ValueError(f"❌ Error creating Grad-CAM model: {e}")
    
    def compute_heatmap(self, image, class_idx=None):
        """
        Compute Grad-CAM heatmap - FIXED
        
        Args:
            image: Preprocessed image (1, H, W, C)
            class_idx: Target class index (uses prediction if None)
            
        Returns:
            heatmap: Normalized heatmap
            predictions: Model predictions
            pred_class: Predicted class index
        """
        # Get predictions
        predictions = self.model.predict(image, verbose=0)
        
        # Use predicted class if not specified
        if class_idx is None:
            class_idx = np.argmax(predictions[0])
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, preds = self.grad_model(image)
            class_channel = preds[:, class_idx]
        
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Handle None gradients
        if grads is None:
            print("⚠️  Warning: Gradients are None, returning blank heatmap")
            return np.zeros((image.shape[1], image.shape[2])), predictions, class_idx
        
        # Global average pooling - FIXED
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU and normalize - FIXED
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        
        if max_val > 0:
            heatmap = heatmap / max_val
        else:
            print("⚠️  Warning: Heatmap has no positive values")
            heatmap = tf.zeros_like(heatmap)
        
        return heatmap.numpy(), predictions, class_idx
    
    def create_visualization(self, image_path, save_path=None, target_class=None):
        """
        Create Grad-CAM visualization for a single image - FIXED
        
        Args:
            image_path: Path to X-ray image
            save_path: Where to save output
            target_class: Specific class to visualize (None = predicted class)
            
        Returns:
            fig: Matplotlib figure
            result_dict: Dictionary with predictions and confidence
        """
        # Load and preprocess image - FIXED
        try:
            original = cv2.imread(str(image_path))
            if original is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")
        
        # Resize for model - MATCHES TRAINING SIZE
        img_resized = cv2.resize(original, (160, 160))
        img_array = img_resized.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Generate heatmap
        heatmap, predictions, pred_class = self.compute_heatmap(img_array, target_class)
        
        # Resize heatmap to original image size
        heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        
        # Apply colormap - FIXED
        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Create overlay - FIXED alpha blending
        overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)
        
        # Create figure - FIXED layout
        fig = plt.figure(figsize=(16, 5))
        
        # Original image
        plt.subplot(1, 4, 1)
        plt.imshow(original)
        plt.title('Original X-Ray', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # Heatmap
        plt.subplot(1, 4, 2)
        plt.imshow(heatmap_resized, cmap='jet', vmin=0, vmax=1)
        plt.title('Activation Map', fontsize=12, fontweight='bold')
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.set_label('Activation Strength', rotation=270, labelpad=15)
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 4, 3)
        plt.imshow(overlay)
        plt.title('Infected Region Highlighted', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # Predictions - FIXED bar chart
        plt.subplot(1, 4, 4)
        colors = ['#4CAF50', '#FF9800', '#F44336']
        y_pos = np.arange(len(self.class_names))
        bars = plt.barh(y_pos, predictions[0], color=colors)
        plt.yticks(y_pos, self.class_names)
        plt.xlabel('Probability', fontsize=11)
        plt.title(
            f'Prediction: {self.class_names[pred_class]}\n'
            f'Confidence: {predictions[0][pred_class]:.2%}',
            fontsize=12, fontweight='bold'
        )
        plt.xlim([0, 1])
        
        # Add percentage labels
        for i, prob in enumerate(predictions[0]):
            plt.text(prob + 0.02, i, f'{prob:.1%}', va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"⚠️  Warning: Could not save to {save_path}: {e}")
                plt.close()
        
        # Prepare result dictionary
        result = {
            'image_path': str(image_path),
            'predicted_class': self.class_names[pred_class],
            'confidence': float(predictions[0][pred_class]),
            'normal_prob': float(predictions[0][0]),
            'bacterial_prob': float(predictions[0][1]),
            'viral_prob': float(predictions[0][2])
        }
        
        return fig, result
    
    def process_folder(self, input_folder, output_folder, max_images=None):
        """
        Process all images in a folder - FIXED
        
        Args:
            input_folder: Folder containing X-ray images
            output_folder: Folder to save Grad-CAM outputs
            max_images: Maximum number of images to process (None = all)
            
        Returns:
            results_df: DataFrame with all predictions and metrics
        """
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all image files - FIXED to find all images
        print(f"\n📂 Scanning for images in: {input_folder}")
        image_files = []
        
        # Search for common image extensions
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
            image_files.extend(list(Path(input_folder).glob(f'**/*{ext}')))
        
        # Remove duplicates and sort
        image_files = sorted(list(set(image_files)))
        
        print(f"✓ Found {len(image_files)} images")
        
        if len(image_files) == 0:
            print(f"⚠️  No images found in {input_folder}")
            return pd.DataFrame()
        
        if max_images:
            image_files = image_files[:max_images]
            print(f"  Processing first {max_images} images")
        
        print(f"\n{'='*60}")
        print(f"Processing {len(image_files)} images")
        print(f"{'='*60}\n")
        
        results = []
        failed = 0
        
        # Process each image with progress bar
        for img_path in tqdm(image_files, desc="Generating Grad-CAM", unit="img"):
            try:
                # Create output filename
                output_filename = f"gradcam_{img_path.name}"
                output_path = os.path.join(output_folder, output_filename)
                
                # Generate visualization
                _, result = self.create_visualization(
                    str(img_path), 
                    save_path=output_path
                )
                
                results.append(result)
                
            except Exception as e:
                failed += 1
                if failed <= 5:  # Only show first 5 errors
                    print(f"\n⚠️  Error processing {img_path.name}: {str(e)}")
                continue
        
        if failed > 5:
            print(f"\n⚠️  {failed - 5} more errors occurred (not shown)")
        
        # Create results DataFrame
        if results:
            results_df = pd.DataFrame(results)
            
            # Save results to CSV
            csv_path = os.path.join(output_folder, 'gradcam_results.csv')
            results_df.to_csv(csv_path, index=False)
            
            print(f"\n✓ Successfully processed {len(results)} images")
            print(f"✓ Results saved to: {csv_path}")
            
            # Print summary statistics
            print(f"\n📊 Prediction Summary:")
            for class_name in self.class_names:
                count = (results_df['predicted_class'] == class_name).sum()
                percentage = (count / len(results_df)) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
            
            return results_df
        else:
            print(f"\n⚠️  No images were successfully processed")
            return pd.DataFrame()
    
    def create_comparison_grid(self, image_paths, save_path, grid_size=(3, 3)):
        """
        Create a grid comparison - FIXED
        
        Args:
            image_paths: List of image paths
            save_path: Where to save the grid
            grid_size: Grid dimensions (rows, cols)
        """
        rows, cols = grid_size
        n_images = min(len(image_paths), rows * cols)
        
        if n_images == 0:
            print("⚠️  No images to create grid")
            return
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
        
        # Handle single subplot case
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        processed = 0
        
        for idx, img_path in enumerate(image_paths[:n_images]):
            try:
                # Load image
                original = cv2.imread(str(img_path))
                if original is None:
                    raise ValueError("Could not load image")
                
                original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                
                # Preprocess
                img_resized = cv2.resize(original, (160, 160))
                img_array = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
                
                # Generate heatmap
                heatmap, predictions, pred_class = self.compute_heatmap(img_array)
                heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
                
                # Create overlay
                heatmap_colored = (plt.cm.jet(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
                overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)
                
                # Plot
                axes[idx].imshow(overlay)
                title = f"{self.class_names[pred_class]}\n{predictions[0][pred_class]:.1%}"
                axes[idx].set_title(title, fontsize=10, fontweight='bold')
                axes[idx].axis('off')
                
                processed += 1
                
            except Exception as e:
                axes[idx].text(0.5, 0.5, f'Error:\n{Path(img_path).name}', 
                              ha='center', va='center', fontsize=8)
                axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Pneumonia Detection - Grad-CAM Grid', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Grid saved to: {save_path} ({processed} images)")
        except Exception as e:
            print(f"⚠️  Could not save grid: {e}")
            plt.close()


def load_model_safe(model_path):
    """
    Safely load model with error handling - FIXED
    """
    print(f"\n🔄 Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    try:
        model = keras.models.load_model(model_path, compile=False)
        print(f"✓ Model loaded successfully")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None


def main():
    """
    Main execution function - FIXED
    """
    print("\n" + "="*70)
    print(" "*15 + "PNEUMONIA GRAD-CAM - BATCH PROCESSOR")
    print("="*70 + "\n")
    
    # ========== CONFIGURATION ==========
    MODEL_PATH = 'pneumonia_model_balanced.h5'
    TRAIN_IMAGES_FOLDER = 'data/train_balanced'
    TEST_IMAGES_FOLDER = 'data/test'
    OUTPUT_TRAIN_FOLDER = 'gradcam_outputs/train_balanced'
    OUTPUT_TEST_FOLDER = 'gradcam_outputs/test'
    
    # Limit images (None = all)
    MAX_TRAIN_IMAGES = 20
    MAX_TEST_IMAGES = 20
    
    # Target layer (None = auto-detect)
    TARGET_LAYER = "conv4_block24_concat"
    # ===================================
    
    print("⚙️  Configuration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Train images: {TRAIN_IMAGES_FOLDER}")
    print(f"  Test images: {TEST_IMAGES_FOLDER}")
    print(f"  Max train images: {MAX_TRAIN_IMAGES if MAX_TRAIN_IMAGES else 'All'}")
    print(f"  Max test images: {MAX_TEST_IMAGES if MAX_TEST_IMAGES else 'All'}")
    
    # Load model
    model = load_model_safe(MODEL_PATH)
    if model is None:
        print("\n❌ Cannot proceed without model")
        print("Please ensure:")
        print("  1. Model file exists: best_model.keras")
        print("  2. Model was trained successfully")
        print("  3. Run: python train_model_optimized.py first")
        return
    
    # Initialize Grad-CAM
    try:
        gradcam = PneumoniaGradCAM(model, layer_name=TARGET_LAYER)
    except Exception as e:
        print(f"\n❌ Error initializing Grad-CAM: {e}")
        return
    
    # Process training images
    train_processed = False
    if os.path.exists(TRAIN_IMAGES_FOLDER):
        print("\n" + "="*70)
        print("📁 PROCESSING TRAINING IMAGES")
        print("="*70)
        
        train_results = gradcam.process_folder(
            TRAIN_IMAGES_FOLDER,
            OUTPUT_TRAIN_FOLDER,
            max_images=MAX_TRAIN_IMAGES
        )
        
        if len(train_results) > 0:
            train_processed = True
            print(f"\n✓ Processed {len(train_results)} training images")
            print(f"✓ Output saved to: {OUTPUT_TRAIN_FOLDER}")
            
            # Create grid visualization
            train_images = list(Path(TRAIN_IMAGES_FOLDER).glob('**/*.jpg'))[:9]
            if not train_images:
                train_images = list(Path(TRAIN_IMAGES_FOLDER).glob('**/*.png'))[:9]
            
            if train_images:
                print("\n📊 Creating grid comparison...")
                gradcam.create_comparison_grid(
                    train_images,
                    os.path.join(OUTPUT_TRAIN_FOLDER, 'grid_comparison.png'),
                    grid_size=(3, 3)
                )
    else:
        print(f"\n⚠️  Training folder not found: {TRAIN_IMAGES_FOLDER}")
    
    # Process test images
    test_processed = False
    if os.path.exists(TEST_IMAGES_FOLDER):
        print("\n" + "="*70)
        print("📁 PROCESSING TEST IMAGES")
        print("="*70)
        
        test_results = gradcam.process_folder(
            TEST_IMAGES_FOLDER,
            OUTPUT_TEST_FOLDER,
            max_images=MAX_TEST_IMAGES
        )
        
        if len(test_results) > 0:
            test_processed = True
            print(f"\n✓ Processed {len(test_results)} test images")
            print(f"✓ Output saved to: {OUTPUT_TEST_FOLDER}")
            
            # Create grid visualization
            test_images = list(Path(TEST_IMAGES_FOLDER).glob('**/*.jpg'))[:9]
            if not test_images:
                test_images = list(Path(TEST_IMAGES_FOLDER).glob('**/*.png'))[:9]
            
            if test_images:
                print("\n📊 Creating grid comparison...")
                gradcam.create_comparison_grid(
                    test_images,
                    os.path.join(OUTPUT_TEST_FOLDER, 'grid_comparison.png'),
                    grid_size=(3, 3)
                )
    else:
        print(f"\n⚠️  Test folder not found: {TEST_IMAGES_FOLDER}")
    
    # Summary
    print("\n" + "="*70)
    print("✓ GRAD-CAM PROCESSING COMPLETE")
    print("="*70)
    
    if train_processed or test_processed:
        print("\n📁 Generated files:")
        if train_processed:
            print(f"  • {OUTPUT_TRAIN_FOLDER}/")
            print(f"    - Individual Grad-CAM images")
            print(f"    - gradcam_results.csv")
            print(f"    - grid_comparison.png")
        if test_processed:
            print(f"  • {OUTPUT_TEST_FOLDER}/")
            print(f"    - Individual Grad-CAM images")
            print(f"    - gradcam_results.csv")
            print(f"    - grid_comparison.png")
    else:
        print("\n⚠️  No images were processed")
        print("Please check:")
        print("  1. Image folders exist and contain images")
        print("  2. Model is properly trained")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()