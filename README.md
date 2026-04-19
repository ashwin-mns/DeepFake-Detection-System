<h1 align="center">🛡️ Enterprise Deepfake Forensics System</h1>

<p align="center">
  <strong>An enterprise-grade web application to detect authentic vs. manipulated media using Deep Learning and CNNs.</strong>
</p>

---

**Live Demonstration:** [https://deepfake-detection-system-er8wlzhlbzdyej7av2syaj.streamlit.app/](https://deepfake-detection-system-er8wlzhlbzdyej7av2syaj.streamlit.app/)

## 📖 About The Project

As synthetic media and "deepfakes" become increasingly hyper-realistic, distinguishing genuine content from artificially generated media is critical for security and trust. This project provides an end-to-end deep learning pipeline to train a Convolutional Neural Network (CNN) for detecting facial manipulations. 

It is bundled with a stunning, modern **Enterprise Command Center UI** featuring live threat analytics, an interactive scanner, system logs, and a local persistence engine to safely store your forensic scan history.

### ✨ Key Features
- **Professional Command Center UI:** A highly polished, corporate dashboard built with Streamlit, custom CSS, and multiple tabs (Scanner, Analytics, Logs, History).
- **Persistent Scan Database:** Automatically records all scans and confidence scores locally into a `.csv` storage system, allowing users to recall and analyze past results.
- **End-to-End Pipeline:** Standardized scripts handling video frame extraction, model training, and real-time inference.
- **Automated Data Splitting:** The training script automatically performs an optimal 80/20 train/test split.
- **Fine-Tuned Sensitivity Control:** Includes an adjustable slider to manually set the threat-detection threshold limit during live scans.

## 🛠️ Built With

* [![TensorFlow][TensorFlow-badge]][TensorFlow-url]
* [![Streamlit][Streamlit-badge]][Streamlit-url]
* [![OpenCV][OpenCV-badge]][OpenCV-url]
* [![NumPy][NumPy-badge]][NumPy-url]

## 📂 Project Structure

```text
Deepfake/
├── .streamlit/
│   └── config.toml         # Dashboard theme configuration
├── dataset/
│   ├── real/               # Directory for authentic training data
│   └── fake/               # Directory for manipulated training data
├── app.py                  # Streamlit Enterprise Command Center & UI components
├── model.py                # CNN architecture and training logic
├── process_videos.py       # Utility to extract frames from .mp4 files
├── requirements.txt        # Python dependencies
├── scan_history.csv        # Automatically generated local database for previous scans
└── model.h5                # Trained Keras model parameters (Generated)
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
To train a highly accurate model, you need a dataset (e.g., FaceForensics++, Kaggle Deepfake Challenge).

1. Ensure the `dataset/real` and `dataset/fake` directories exist.
2. Place your raw `.mp4` video files or `.jpg`/`.png` images inside.

**If using video files:** Extract frames by running the preprocessing script:
```bash
python process_videos.py
```
*(This extracts 3 frames per video to build your image dataset pipeline).*

### 4. Training the Model
Initiate the training process:

```bash
python model.py
```
- Automatically processes images and creates an 80/20 train/validation split.
- Saves learned weights to `model.h5`.

### 5. Running the Command Center
Launch the Streamlit dashboard to interact with your trained model:

```bash
streamlit run app.py
```
Your default browser will natively open to `http://localhost:8501`. 

## 💡 Usage Guide

With the new multi-tab dashboard interface, the system operates across 4 primary layers:

1. **Scanner Tab:** Drag & drop a standard image file (JPG/PNG). Click the **Initialize Scan 🔍** button to trigger the analysis sequence. The pipeline predicts authenticity and confidence ratings.
2. **Scan History Tab:** View a permanent, tabular log of all previously scanned images, including their exact timestamps and detection results.
3. **Analytics Dashboard Tab:** Observe hypothetical global metrics such as system latency, total processing volume, and threat prevention charts.
4. **System Logs Tab:** An emulated server tail log to view backend authentication nodes, spatial dimension extractions, and model synchronizations.

## 🔭 Future Roadmap (Upcoming Enhancements)

- [ ] **Direct Video Analysis:** Allow users to directly upload `.mp4` files into the UI for frame-by-frame temporal analysis.
- [ ] **Explainable AI (XAI) Overlay:** Implement **Grad-CAM heatmaps** to visually highlight *exactly* which pixels or facial regions triggered the deepfake alert.
- [ ] **Automated Face Cropping Engine:** Integrate **MediaPipe / MTCNN** to automatically isolate the human face in an image before running it through the CNN logic.
- [ ] **Export to PDF Security Reports:** A new button to dynamically generate and download a branded PDF file certifying the authenticity results of a given document.
- [ ] **Bulk/Batch Operations:** Facilitate the bulk upload of `.zip` files allowing researchers to analyze thousands of files silently in the background.

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
