import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from heatmap_utils import get_heatmap, overlay_heatmap 
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pneumonia AI Diagnostic", page_icon="🫁", layout="wide")

st.title("🫁 AI-Assisted Pneumonia Detection")
st.markdown("**Status:** System Ready | **Model:** ResNet50 High-Recall")

# --- SIDEBAR ---
st.sidebar.header("Diagnostic Controls")
# High Recall means we set a LOW threshold (e.g., 30%) to catch everything
confidence_threshold = st.sidebar.slider("Sensitivity Threshold", 0.0, 1.0, 0.30, 0.05)
st.sidebar.info("Lower threshold = Higher Sensitivity (More likely to flag Pneumonia)")

# --- LOAD MODEL ---
@st.cache_resource
def load_pneumonia_model():
    if os.path.exists('best_model.keras'):
        return load_model('best_model.keras')
    return None

model = load_pneumonia_model()

if model is None:
    st.error("⚠️ Model file 'best_model.keras' not found. Please train the model first.")
else:
    # --- FILE UPLOAD ---
    uploaded_file = st.file_uploader("Upload Chest X-Ray (JPEG/PNG)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # FIX 1: Reset file pointer to be safe
        uploaded_file.seek(0)
        
        # FIX 2: Read bytes directly from memory
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, 1) # 1 means load as color

        # Check if image loaded correctly
        if original_image is None:
            st.error("Error loading image. The file might be corrupted or not a valid image format.")
        else:
            # --- SUCCESS BLOCK (Everything happens INSIDE here) ---
            
            # Convert BGR to RGB (OpenCV loads as BGR by default)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Resize for Model
            IMG_SIZE = 128
            img_processed = cv2.resize(original_image, (IMG_SIZE, IMG_SIZE))
            
            # Create batch: (1, 128, 128, 3)
            img_batch = np.expand_dims(img_processed / 255.0, axis=0).astype('float32')

            # --- PREDICTION ---
            with st.spinner('Analyzing lung patterns...'):
                preds = model.predict(img_batch)
                
            # Get Probabilities: [Normal, Bacterial, Viral]
            prob_normal = preds[0][0]
            prob_bacterial = preds[0][1]
            prob_viral = preds[0][2]
            
            # Logic: If (Bacterial + Viral) > Threshold -> Flag it
            total_pneumonia_prob = prob_bacterial + prob_viral
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(original_image, caption="Original X-Ray", use_container_width=True)

            with col2:
                if total_pneumonia_prob > confidence_threshold:
                    # DETECTED
                    class_idx = 1 if prob_bacterial > prob_viral else 2
                    label = "BACTERIAL PNEUMONIA" if class_idx == 1 else "VIRAL PNEUMONIA"
                    confidence = max(prob_bacterial, prob_viral)
                    
                    st.error(f"🚨 **DETECTED: {label}**")
                    st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Generate Heatmap
                    try:
                        heatmap = get_heatmap(model, img_batch, class_idx)
                        overlay = overlay_heatmap(img_processed, heatmap, alpha=0.5)
                        st.image(overlay, caption="Class Activation Map (Red = Infection Focus)", use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate heatmap: {e}")
                    
                else:
                    # NORMAL
                    st.success("✅ **RESULT: NORMAL**")
                    st.metric("Confidence", f"{prob_normal:.2%}")
                    st.info("No significant infection patterns detected.")