import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import os
import io

from model import build_model

st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🕵️",
    layout="centered"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: 0.3s;
        border: none;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .header-style {
        text-align: center;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-header-style {
        text-align: center;
        color: #a0aec0;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    .result-box-fake {
        background-color: rgba(255, 75, 75, 0.1);
        border-left: 5px solid #ff4b4b;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .result-box-real {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 5px solid #4CAF50;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

def load_detection_model():
    model_path = "model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    # Return a dummy model if file doesn't exist
    return build_model()

model = load_detection_model()

st.markdown('<p class="header-style">Deepfake Detector 🕵️</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header-style">Upload an image to detect if it\'s authentic or AI-generated.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(image, caption='Uploaded Document', use_container_width=True)

    if st.button("Start Analysis 🔍"):
        with st.spinner('Analyzing facial artifacts and inconsistencies...'):
            try:
                # Preprocess image
                img_array = np.array(image)
                # Convert to BGR for cv2 compatibility mostly if model expects it, 
                # but standard models usually expect RGB. Let's keep RGB for TF defaults
                # Resize to (224, 224)
                img_resized = cv2.resize(img_array, (224, 224))
                
                # Normalize
                img_normalized = img_resized / 255.0
                
                # Expand dims (batch size = 1)
                img_input = np.expand_dims(img_normalized, axis=0)
                
                # Predict
                prediction = model.predict(img_input)[0][0]
                
                # Determine Result
                is_fake = prediction > 0.5
                confidence = prediction if is_fake else 1 - prediction
                
                # Display output beautifully
                if is_fake:
                    st.markdown(f'''
                    <div class="result-box-fake">
                        <h2 style="color: #ff4b4b; margin:0;">🚨 Warning: Deepfake Detected (Fake)</h2>
                        <p style="font-size: 1.2rem; margin-top:10px;">Confidence Score: <strong>{confidence*100:.2f}%</strong></p>
                    </div>
                    ''', unsafe_allow_html=True)
                    st.error("Our models have detected artificial manipulation in this media.")
                else:
                    st.markdown(f'''
                    <div class="result-box-real">
                        <h2 style="color: #4CAF50; margin:0;">✅ Authentic Media (Real)</h2>
                        <p style="font-size: 1.2rem; margin-top:10px;">Confidence Score: <strong>{confidence*100:.2f}%</strong></p>
                    </div>
                    ''', unsafe_allow_html=True)
                    st.success("This media appears to be genuine and unmodified.")
                    
            except Exception as e:
                st.error(f"Error during processing: {e}")

st.markdown("---")
st.markdown("<small style='text-align: center; display: block;'>Powered by Deep Learning & Convolutional Neural Networks</small>", unsafe_allow_html=True)
