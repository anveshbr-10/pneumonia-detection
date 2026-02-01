import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI-Assisted Pneumonia Diagnostics",
    page_icon="🫁",
    layout="wide"
)

# ------------------ SIDEBAR ------------------
st.sidebar.title("🩺 System Info")
st.sidebar.markdown(
    """
    **Model:** DenseNet-121  
    **Task:** Multi-class Pneumonia Detection  
    **Classes:**  
    - Normal  
    - Bacterial Pneumonia  
    - Viral Pneumonia  

    ⚠️ *Decision-support tool, not a replacement for clinicians.*
    """
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.4,
    max_value=0.8,
    value=0.55,
    step=0.05
)

# ------------------ MAIN TITLE ------------------
st.title(" AI-Assisted Pneumonia Diagnostics")
st.caption(
    "Upload a chest X-ray to receive an AI-assisted prediction with visual explanation (Grad-CAM)."
)

st.divider()

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pneumonia_model_balanced.h5")

model = load_model()

class_names = ['NORMAL', 'BACTERIAL_PNEUMONIA', 'VIRAL_PNEUMONIA']
IMG_SIZE = 160
LAST_CONV_LAYER = "conv4_block24_concat"

# ------------------ GRAD-CAM ------------------
def generate_gradcam(img_array):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(LAST_CONV_LAYER).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap, int(class_idx), predictions.numpy()

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # -------- IMAGE PREP --------
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------- PREDICTION --------
    heatmap, pred_idx, preds = generate_gradcam(img_array)
    confidence = preds[0][pred_idx]
    # ----- CREATE OVERLAY (DO NOT MOVE BELOW DISPLAY) -----
    original = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

    # RED-only CAM (infected area)
    heatmap_norm = np.uint8(255 * heatmap_resized)
    red_mask = np.zeros_like(original)
    red_mask[:, :, 0] = heatmap_norm
    red_mask[heatmap_norm < 120] = 0  # suppress weak activations

    overlay = cv2.addWeighted(original, 0.85, red_mask, 0.45, 0)

    if confidence < confidence_threshold:
        prediction = "Uncertain – Needs Doctor Review"
        status_color = "⚠️"
    else:
        prediction = class_names[pred_idx]
        status_color = "✅"

    # ------------------ LAYOUT ------------------
    col1, col2 = st.columns([1, 1])

    # -------- LEFT: IMAGE --------
    with col1:
        st.subheader("🖼️ Uploaded X-ray")
        st.image(img, width=350)

    # -------- RIGHT: RESULTS --------
    with col2:
        st.subheader("📊 Prediction Result")

        st.markdown(
            f"""
            ### {status_color} **{prediction}**
            **Confidence:** `{confidence*100:.2f}%`
            """
        )

        # Probability bar chart
        prob_df = pd.DataFrame({
            "Class": class_names,
            "Probability": preds[0]
        })

        st.bar_chart(
            prob_df.set_index("Class"),
            height=220
        )

    st.divider()



    st.markdown("### 🔥 Infected Region Highlight (CAM)")
    cam_col1, cam_col2, cam_col3 = st.columns([1, 2, 1])
    with cam_col2:
        st.image(
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            caption="Red regions indicate infected lung areas",
            width=350   # 👈 THIS is the key
            )
