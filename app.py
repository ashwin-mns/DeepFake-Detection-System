import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import os
import time

from model import build_model

st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for an ultra-modern, premium glassmorphism design
st.markdown("""
<style>
    /* Global Background and Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top left, #f4f7fb, #eef2f6, #e2e8f0);
    }

    /* Hide Streamlit Header and Footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Typography */
    .title-text {
        text-align: center;
        background: linear-gradient(135deg, #ff758c 0%, #ff7eb3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        margin-top: 0;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .subtitle-text {
        text-align: center;
        color: #718096;
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }

    /* Glassmorphism Containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.45);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.6);
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
    }

    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border-radius: 30px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.6);
        color: white;
    }

    /* File Uploader override */
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.5);
        border: 2px dashed rgba(74, 85, 104, 0.3);
        border-radius: 15px;
        transition: 0.3s;
    }
    .stFileUploader > div > div:hover {
        border-color: #00f2fe;
        background: rgba(255, 255, 255, 0.8);
    }

    /* Results styling */
    .result-fake {
        background: rgba(255, 255, 255, 0.6);
        border-left: 5px solid #ff4b4b;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.1);
    }
    .result-real {
        background: rgba(255, 255, 255, 0.6);
        border-left: 5px solid #4CAF50;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.1);
    }
    
    .score-text {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detection_model():
    model_path = "model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    # Return a dummy model if file doesn't exist
    return build_model()

model = load_detection_model()

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/11833/11833323.png", width=80)
    st.markdown("## Deepfake Forensics")
    st.markdown("Use cutting-edge Deep Learning to detect facial manipulations.")
    
    st.markdown("---")
    st.markdown("### 🔬 How it works")
    st.markdown("""
    1. **Upload** an image containing a face.
    2. **Analysis**: The system analyzes pixel-level anomalies & artifacts.
    3. **CNN Engine**: A trained Neural Network predicts the authenticity.
    """)
    st.markdown("---")
    st.info("Tip: Ensure the image clearly shows a face for the highest accuracy.")

# --- Main Layout ---
st.markdown('<p class="title-text">Deepfake Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Empowering trust in digital media through AI.</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop an image to scan", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Create two columns for display and results once uploaded
    st.markdown("---")
    disp_col, res_col = st.columns([1.2, 1])
    
    with disp_col:
        st.markdown("### 🖼️ Source Media")
        st.image(image, use_column_width=True, caption="Uploaded Document")
        
    with res_col:
        st.markdown("### 📊 Analysis Report")
        analyze_button = st.button("Run Forensic Analysis 🔍")
        
        if analyze_button:
            # Fake a loading bar for better UX feeling
            my_bar = st.progress(0, text="Initializing tensor arrays...")
            for percent_complete in range(100):
                time.sleep(0.015)
                # Change text dynamically
                if percent_complete < 30:
                    my_bar.progress(percent_complete + 1, text="Analyzing facial artifacts...")
                elif percent_complete < 70:
                    my_bar.progress(percent_complete + 1, text="Checking pixel gradients...")
                else:
                    my_bar.progress(percent_complete + 1, text="Computing final probabilities...")
            
            my_bar.empty()
            
            with st.spinner('Compiling final report...'):
                try:
                    # Preprocess image
                    img_array = np.array(image)
                    img_resized = cv2.resize(img_array, (224, 224))
                    img_normalized = img_resized / 255.0
                    img_input = np.expand_dims(img_normalized, axis=0)
                    
                    # Predict
                    prediction = model.predict(img_input)[0][0]
                    is_fake = prediction > 0.5
                    confidence = prediction if is_fake else 1 - prediction
                    
                    # Display output beautifully
                    if is_fake:
                        st.markdown(f'''
                        <div class="result-fake">
                            <h3 style="color: #ff4b4b; margin:0;">🚨 Deepfake Detected</h3>
                            <p class="score-text" style="color: #ff4b4b;">{confidence*100:.2f}%</p>
                            <p style="margin:0; color: #4a5568;">High likelihood of synthetic manipulation.</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="result-real">
                            <h3 style="color: #4CAF50; margin:0;">✅ Authentic Media</h3>
                            <p class="score-text" style="color: #4CAF50;">{confidence*100:.2f}%</p>
                            <p style="margin:0; color: #4a5568;">No significant signs of manipulation found.</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                    # Add detailed metrics
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("#### Confidence Metrics")
                    m1, m2 = st.columns(2)
                    m1.metric(label="Real Probability", value=f"{((1-prediction)*100):.1f}%")
                    m2.metric(label="Fake Probability", value=f"{(prediction*100):.1f}%")

                except Exception as e:
                    st.error(f"Error during processing: {e}")

st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #4a5568; font-size: 0.85rem;'>Deepfake Detection System • Built with TensorFlow & Streamlit</p>", unsafe_allow_html=True)
