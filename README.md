# Real-Time Facial Emotion Detection Using Deep Learning

A production-quality deep learning project for facial expression analysis and emotion recognition. The application uses a custom-built Convolutional Neural Network (CNN) trained with TensorFlow/Keras and uses OpenCV with Haar Cascades to perform face detection and classification in real time from a camera feed or single images.

Four primary emotions are supported:
- **Happy**
- **Sad**
- **Angry**
- **Surprised**

---

## Project Structure

```text
Emotion-Detection/
│
├── dataset/                         # Dataset directory (created automatically)
│   ├── train/                       # Directory containing subfolders for each emotion class
│   └── test/                        # Directory containing subfolders for each emotion class
│
├── models/                          # Storage directory for model artifacts
│   ├── emotion_model.keras          # Trained weights and neural network topology (saved model)
│   └── training_history.png         # Visualization of loss and accuracy metrics
│
├── haarcascade/                     # Cascade classifiers
│   └── haarcascade_frontalface_default.xml # Face detection parameters (downloaded automatically)
│
├── train.py                         # Defines CNN architecture and executes model training
├── predict.py                       # Runs inference on a single static image file
├── camera.py                        # Launches live webcam feed for real-time inference
├── utils.py                         # Handles directory setup, Haar cascade fetching, and data generation
├── requirements.txt                 # Manifest of libraries and system requirements
├── README.md                        # Documentation and project manual
└── .gitignore                       # System and build configuration file ignore rules
```

---

## Features

1. **Self-Contained Data Pipeline**: If no external images are present in the `dataset/` directory, the training script automatically creates a custom synthetic grayscale dataset with geometric representations of the facial expressions. This makes the project immediately runnable and testable.
2. **Deep CNN Architecture**: Implements an optimized multi-layered CNN structure with Batch Normalization, Max Pooling, and Dropout regularization.
3. **Robust Real-Time Detection**: Incorporates OpenCV Haar Cascades for face locating and crops region-of-interests (ROIs) dynamically for prediction.
4. **Single-Image Diagnostics**: Supports prediction CLI outputs containing confidence percentages overlaid on bounding boxes.

---

## Installation

### Prerequisites
- Python 3.11.x
- Webcam (for `camera.py`)

### 1. Clone or Copy the Project
Ensure all files are arranged within the `Emotion-Detection/` workspace.

### 2. Set Up a Virtual Environment (Recommended)
Navigate to the project root directory and create a virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.venv\Scripts\activate.bat
# On macOS / Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1: Train the CNN Model
Start the training script to configure datasets and train the network:

```bash
python train.py
```

*Note: Running `train.py` automatically checks for dataset directory structures. If they do not exist, it downloads the Haar Cascade model and generates synthetic samples to ensure the training loop operates out-of-the-box. If you want to use a custom dataset (such as the FER2013 dataset), organize your images as follows:*
- Place training images inside `dataset/train/{happy,sad,angry,surprised}/`
- Place testing/validation images inside `dataset/test/{happy,sad,angry,surprised}/`

Once training finishes:
- The optimal model is saved as `models/emotion_model.keras`.
- Training curves are saved as `models/training_history.png`.

### Step 2: Run Real-Time Webcam Detection
To run real-time predictions via webcam, run:

```bash
python camera.py
```
- A window will display your camera feed.
- Faces will be annotated with green bounding boxes and predicted emotions + confidence scores.
- **Press `q` to exit** the live stream.

### Step 3: Run Inference on a Single Image
To classify the emotion present in a single static photo, run the prediction script:

```bash
python predict.py --image path/to/your/image.jpg
```
- The script detects any faces in the image, runs the classifier on each, draws labels, and saves the output to a new image (e.g., `image_predicted.jpg` in the local directory).
- If no faces are found, it runs the classifier over the entire image as a fallback.

---

## Model Architecture Details
The CNN consists of:
- **3 Convolutional blocks**:
  - Double convolutional layers with 3x3 kernels and ReLU activation (32, 64, and 128 filters respectively).
  - Batch normalization after each convolution for faster convergence.
  - MaxPooling layers with 2x2 window size.
  - Dropout layers (25%) to combat overfitting.
- **Dense fully-connected layer**:
  - Flattening.
  - Two dense layers (512 and 256 units respectively) with Dropout (50%).
  - Softmax classifier output mapping to the 4 target emotions.
