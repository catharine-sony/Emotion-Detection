"""
Real-time webcam inference script for Emotion Detection.
Opens the webcam feed, runs Haar Cascade face detection, classifies the facial expression,
and overlays bounding boxes and classification labels onto the video stream.
"""

import os
import sys
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

def main():
    # 1. Verify and load the trained CNN model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model file not found at '{MODEL_PATH}'.")
        print("Please train the model first by running: python train.py")
        sys.exit(1)
        
    print(f"Loading trained emotion model from {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # 2. Verify and load Haar Cascade XML
    if not os.path.exists(HAAR_FILE_PATH):
        print(f"Error: Haar Cascade XML file not found at '{HAAR_FILE_PATH}'.")
        print("Please ensure you run train.py first to auto-download standard resources.")
        sys.exit(1)
        
    face_cascade = cv2.CascadeClassifier(HAAR_FILE_PATH)
    if face_cascade.empty():
        print("Error: Could not load Cascade Classifier. XML file may be corrupted.")
        sys.exit(1)
        
    # 3. Initialize Webcam
    print("Initializing camera stream...")
    # Try index 0, which is standard for built-in or external webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam at camera index 0.")
        print("If you are using a different camera, check its connection or index.")
        sys.exit(1)
        
    print("\n-----------------------------------------------------")
    print("Camera interface started successfully!")
    print("Instructions:")
    print(" - Look directly at the camera.")
    print(" - Press 'q' key in the camera window to exit.")
    print("-----------------------------------------------------\n")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera. Exiting...")
            break
            
        # Convert frame to grayscale for face detection and preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in faces:
            # Draw green bounding box around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Crop the face ROI (Region of Interest)
            roi_gray = gray[y:y+h, x:x+w]
            
            # Preprocess the cropped face ROI
            # 1. Resize to target dimension 48x48
            roi_resized = cv2.resize(roi_gray, IMAGE_SIZE)
            
            # 2. Normalize values between [0, 1]
            roi_normalized = roi_resized / 255.0
            
            # 3. Reshape to fit model expectations: (batch_size, height, width, channels)
            roi_reshaped = np.reshape(roi_normalized, (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
            
            # Run inference
            predictions = model.predict(roi_reshaped, verbose=0)[0]
            max_idx = np.argmax(predictions)
            emotion = CLASSES[max_idx]
            confidence = predictions[max_idx] * 100
            
            # Draw classification text above the bounding box
            label = f"{emotion} ({confidence:.1f}%)"
            
            # Formatting variables for bounding box labels
            # Ensure text labels stay inside the window boundaries
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            y_text = max(y - 10, label_height + 10)
            
            # Draw dark background label box for premium look and high contrast
            cv2.rectangle(
                frame,
                (x, y_text - label_height - 5),
                (x + label_width, y_text + baseline - 5),
                (0, 255, 0),
                cv2.FILLED
            )
            
            # Write text label onto the frame
            cv2.putText(
                frame,
                label,
                (x, y_text - 5),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                lineType=cv2.LINE_AA
            )
            
        # Display the resulting frame in a window
        cv2.imshow('Real-Time Facial Emotion Detection', frame)
        
        # Monitor keystrokes: Quit if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit command received. Shutting down...")
            break
            
    # Cleanup camera resource and close window instances
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed released and windows destroyed successfully.")

if __name__ == "__main__":
    main()
