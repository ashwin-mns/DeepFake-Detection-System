<h1 align="center">🕵️ Deepfake Detection System</h1>

<p align="center">
  <strong>A premium, light-themed web application to detect authentic vs. manipulated media using Deep Learning.</strong>
</p>

---

https://deepfake-detection-system-er8wlzhlbzdyej7av2syaj.streamlit.app/

## 📖 About The Project

As synthetic media and "deepfakes" become increasingly hyper-realistic, distinguishing genuine content from artificially generated media is harder than ever. This project provides an end-to-end deep learning pipeline to train a Convolutional Neural Network (CNN) for detecting facial manipulations, bundled with a stunning, modern **glassmorphic light-theme Web UI** for seamless user interaction.

### ✨ Key Features
- **Premium Glassmorphic UI:** A beautifully designed, vibrant, light-themed interface built with Streamlit, custom CSS, and modern visual aesthetics.
- **End-to-End Pipeline:** Standardized scripts handling video frame extraction, model training, and real-time inference.
- **Automated Data Splitting:** The training script automatically performs an optimal 80/20 train/test split.
- **Smart Fallback Engine:** If an extensive dataset is missing, the system auto-generates a compiled placeholder model so you can still launch and explore the UI.
- **Comprehensive Analysis Dashboard:** The application provides intuitive visual cues, binary classification (Authentic vs. Fake), and a detailed probability breakdown.

## 🛠️ Built With

* [![TensorFlow][TensorFlow-badge]][TensorFlow-url]
* [![Streamlit][Streamlit-badge]][Streamlit-url]
* [![OpenCV][OpenCV-badge]][OpenCV-url]
* [![NumPy][NumPy-badge]][NumPy-url]

## 📂 Project Structure

```text
Deepfake/
├── .streamlit/
│   └── config.toml         # Forces Streamlit's premium light theme
├── dataset/
│   ├── real/               # Directory for authentic images/videos
│   └── fake/               # Directory for manipulated images/videos
├── app.py                  # Streamlit web application & UI components
├── model.py                # CNN architecture and training logic
├── process_videos.py       # Utility to extract frames from .mp4 files
├── requirements.txt        # Python dependencies
└── model.h5                # Trained Keras model (Generated)
```

## 🚀 Getting Started

Follow these steps to set up the project locally on your machine.

### 1. Prerequisites
Ensure you have Python 3.8 or higher installed on your system.

### 2. Installation
Navigate to your project directory and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Preparing the Dataset
To train a highly accurate model, you need a dataset (e.g., FaceForensics++, Kaggle Deepfake Challenge dataset).

1. Ensure the `dataset/real` and `dataset/fake` directories exist in the project root.
2. Place your raw `.mp4` video files or `.jpg`/`.png` images inside their respective folders.

**If using video files:** Extract frames by running the preprocessing script:
```bash
python process_videos.py
```
*(This extracts 3 frames per video to build your image dataset pipeline).*

### 4. Training the Model
Once your data is populated, initiate the training process:

```bash
python model.py
```
- The script automatically processes the internal images and yields an 80/20 train/validation split.
- It trains a custom CNN for 5 epochs.
- It evaluates the testing dataset and saves the learned weights to `model.h5`.
- *(Note: If no dataset is found, it compiles an untrained model to allow UI testing).*

### 5. Running the Web UI
Launch the sleek Streamlit dashboard to interact with your trained model:

```bash
streamlit run app.py
```
Your default browser will natively open to `http://localhost:8501`. Upload an image containing a face and run the forensic analysis!

## 💡 Usage Guide

1. **Upload Media:** Drag and drop a standard image file (JPG/PNG) into the designated glassmorphic drop zone.
2. **Review Upload:** The source image will be displayed on the left pane of the analysis layout.
3. **Run Forensics:** Click the **"Run Forensic Analysis 🔍"** button.
4. **View Results:** The system intelligently simulates deep artifact analysis and presents a clear **Authentic** or **Deepfake Detected** categorization along with mathematically computed confidence metrics.

## 🔭 Future Roadmap

- [ ] Implement advanced architectures like **ResNet50** or **EfficientNet** for superior feature extraction.
- [ ] Integrate a dedicated face-cropping pipeline (e.g., MTCNN, MediaPipe) before sending images to the classification model.
- [ ] Support direct video `.mp4` uploads and frame-by-frame temporal analysis in the web app.
- [ ] Add explainability overlays (e.g., Grad-CAM) to intuitively highlight manipulated regions of the face.

---
<p align="center">
  <i>Developed with ❤️ using Python & Deep Learning</i>
</p>

<!-- MARKDOWN LINKS & IMAGES -->
[TensorFlow-badge]: https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white
[TensorFlow-url]: https://www.tensorflow.org/
[Streamlit-badge]: https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white
[Streamlit-url]: https://streamlit.io/
[OpenCV-badge]: https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white
[OpenCV-url]: https://opencv.org/
[NumPy-badge]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
