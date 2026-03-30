# Deepfake Detection System 🕵️

A complete deep learning pipeline and web application that detects whether an image or video is authentic (real) or artificially manipulated (deepfake). The system uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras and features a modern, user-friendly interface powered by Streamlit.

## 🌟 Features

- **Media Preprocessing**: Built-in script (`process_videos.py`) to extract frames from video files (.mp4), converting them into an image dataset.
- **Deep Learning Model**: A custom CNN architecture (`model.py`) optimized for binary classification (Real vs. Fake), with automatic 80/20 train/test splitting.
- **Interactive UI**: A vibrant, modern web interface (`app.py`) built with Streamlit allowing users to upload images and receive instant predictions with confidence scores.
- **Dummy Mode**: Auto-generates an untrained placeholder model so the application can be run and tested even before a full dataset is acquired.

## 📂 Project Structure

- `app.py`: The Streamlit web application for real-time inference and user interaction.
- `model.py`: Handles checking the dataset, building the CNN architecture, and training the model.
- `process_videos.py`: Utility script to process raw `.mp4` video files into image frames for training.
- `requirements.txt`: Contains all necessary Python dependencies.
- `dataset/` (Generated): Directory where you should place your real and fake data inside `dataset/real` and `dataset/fake`.
- `model.h5` (Generated): The trained TensorFlow/Keras model.

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Navigate to this project directory, and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Preparing the Dataset
To train the model properly, you need a dataset (like FaceForensics++ or Kaggle Deepfake datasets). 
1. Create a `dataset` folder in the root directory if it doesn't exist.
2. Inside `dataset`, create two subfolders: `real` and `fake`.
3. Place your raw `.mp4` video files or `.jpg`/.`png` images inside their respective directories.

If you are using video files, extract frames by running:
```bash
python process_videos.py
```
*(This extracts 3 frames per video to build your image dataset)*

### 3. Training the Model
Once your `dataset/real` and `dataset/fake` folders contain images, run the training script:

```bash
python model.py
```
This script will automatically detect the images, perform an explicit 80/20 train/test split, train the CNN model for 5 epochs, and save the final model as `model.h5`. 
*(Note: If no images are found, it generates an untrained dummy model so you can still preview the UI).*

### 4. Running the Web App
Start the Streamlit application to use the model:

```bash
python -m streamlit run app.py
```
This will open the web interface in your default browser. Upload any image to evaluate its authenticity!

## 🛠️ Built With

- **[TensorFlow / Keras](https://www.tensorflow.org/)**: Core deep learning framework driving model architecture and training.
- **[OpenCV](https://opencv.org/)**: Utilized for video preprocessing and frame extraction.
- **[Streamlit](https://streamlit.io/)**: Provides the responsive and aesthetic web UI frontend.
- **[Pillow & NumPy](https://numpy.org/)**: Handling image matrix transformations and array mathematics.

## 📝 Notes

- Ensure your dataset is roughly balanced (an equivalent number of real and fake images) for optimal model accuracy and to prevent bias.
- For a more robust production model, you may consider modifying `model.py` to utilize transfer learning with models like **ResNet** or **EfficientNet**, or injecting a dedicated face-extraction library.
