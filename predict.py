"""
Inference script to predict facial emotions from a single image file.
Detects faces using Haar Cascade, processes them, runs the CNN classifier,
draws bounding boxes with predicted labels, and saves the output image.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Constants
IMAGE_SIZE = (48, 48)
CLASSES = ["angry", "happy", "sad", "surprised"]

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAAR_FILE_PATH = os.path.join(BASE_DIR, 'haarcascade', 'haarcascade_frontalface_default.xml')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'emotion_model.keras')

def load_emotion_model():
    """
    Loads the saved Keras emotion recognition model.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'.")
        print("Please train the model first by running: python train.py")
        sys.exit(1)
        
    print(f"Loading trained model from {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def load_face_cascade():
    """
    Loads the Haar Cascade face detector XML.
    """
    if not os.path.exists(HAAR_FILE_PATH):
        print(f"Error: Haar Cascade XML not found at '{HAAR_FILE_PATH}'.")
        print("Please run train.py or ensure the xml file is downloaded.")
        sys.exit(1)
        
    face_cascade = cv2.CascadeClassifier(HAAR_FILE_PATH)
    return face_cascade

def predict_image(image_path, model, face_cascade, output_path=None):
    """
    Runs face detection and emotion prediction on a single image.
    """
    if not os.path.exists(image_path):
        print(f"Error: Input image file does not exist at '{image_path}'")
        return
        
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at '{image_path}'")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    print(f"Detected {len(faces)} face(s) in the image.")
    
    # If no faces were detected, process the entire image as fallback
    if len(faces) == 0:
        print("Warning: No faces detected. Classifying the entire image as fallback...")
        resized = cv2.resize(gray, IMAGE_SIZE)
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
        
        predictions = model.predict(reshaped, verbose=0)[0]
        max_idx = np.argmax(predictions)
        emotion = CLASSES[max_idx]
        confidence = predictions[max_idx] * 100
        
        print(f"Predicted emotion (entire image): {emotion} ({confidence:.2f}%)")
        
        # Write predictions on image
        cv2.putText(img, f"Fallback: {emotion} ({confidence:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        for (x, y, w, h) in faces:
            # Crop the detected face region
            roi_gray = gray[y:y+h, x:x+w]
            
            # Preprocess the cropped face image
            roi_resized = cv2.resize(roi_gray, IMAGE_SIZE)
            roi_normalized = roi_resized / 255.0
            roi_reshaped = np.reshape(roi_normalized, (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
            
            # Predict
            predictions = model.predict(roi_reshaped, verbose=0)[0]
            max_idx = np.argmax(predictions)
            emotion = CLASSES[max_idx]
            confidence = predictions[max_idx] * 100
            
            print(f"Detected face at [{x}, {y}, {w}, {h}]: {emotion} ({confidence:.2f}%)")
            
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw label background
            label = f"{emotion}: {confidence:.1f}%"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            y_label_start = max(y - 10, label_height + 10)
            
            cv2.rectangle(
                img,
                (x, y_label_start - label_height - 5),
                (x + label_width, y_label_start + baseline - 5),
                (0, 255, 0),
                cv2.FILLED
            )
            
            # Draw label text
            cv2.putText(
                img, label, (x, y_label_start - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )
            
    # Save the output image
    if output_path is None:
        filename, ext = os.path.splitext(os.path.basename(image_path))
        output_path = os.path.join(BASE_DIR, f"{filename}_predicted{ext}")
        
    cv2.imwrite(output_path, img)
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify emotions in a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file.")
    parser.add_argument("--output", type=str, default=None, help="Path to save prediction image. (Optional)")
    args = parser.parse_args()
    
    # Load model and cascade
    model = load_emotion_model()
    face_cascade = load_face_cascade()
    
    # Perform prediction
    predict_image(args.image, model, face_cascade, args.output)
