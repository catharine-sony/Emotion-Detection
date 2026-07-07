"""
Utility functions for the Emotion Detection project.
Includes Haar Cascade downloader and a Synthetic Dataset Generator for out-of-the-box training and testing.
"""

import os
import urllib.request
import numpy as np
import cv2

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAAR_DIR = os.path.join(BASE_DIR, 'haarcascade')
HAAR_FILE_PATH = os.path.join(HAAR_DIR, 'haarcascade_frontalface_default.xml')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

def download_haar_cascade():
    """
    Downloads the Haar Cascade XML file for frontal face detection
    from OpenCV's official repository if it does not exist locally.
    """
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    
    if not os.path.exists(HAAR_DIR):
        os.makedirs(HAAR_DIR)
        print(f"Created directory: {HAAR_DIR}")
        
    if not os.path.exists(HAAR_FILE_PATH):
        print("Downloading Haar Cascade XML from OpenCV repository...")
        try:
            urllib.request.urlretrieve(url, HAAR_FILE_PATH)
            print(f"Successfully downloaded Haar Cascade to {HAAR_FILE_PATH}")
        except Exception as e:
            print(f"Error downloading Haar Cascade: {e}")
            raise e
    else:
        print("Haar Cascade XML already exists.")

def draw_face_features(emotion, size=48):
    """
    Generates a 48x48 grayscale image representing a simplified face expression.
    """
    # Create black canvas
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Add random background noise
    noise = np.random.normal(30, 10, (size, size)).astype(np.float32)
    img = cv2.addWeighted(img, 0.5, noise.astype(np.uint8), 0.5, 0)
    
    # Draw eyes (constant for all emotions to make model focus on mouth/eyebrows)
    # Left eye
    cv2.circle(img, (16, 18), 3, 200, -1)
    # Right eye
    cv2.circle(img, (32, 18), 3, 200, -1)
    
    if emotion == "happy":
        # Smiling mouth (ellipse arc)
        cv2.ellipse(img, (24, 28), (10, 6), 0, 0, 180, 220, 2)
    elif emotion == "sad":
        # Frowning mouth (ellipse arc inverted)
        cv2.ellipse(img, (24, 34), (10, 6), 0, 180, 360, 220, 2)
    elif emotion == "angry":
        # Straight mouth
        cv2.line(img, (16, 32), (32, 32), 220, 2)
        # Slanted eyebrows
        cv2.line(img, (12, 12), (18, 16), 220, 2) # Left eyebrow slanting down
        cv2.line(img, (36, 12), (30, 16), 220, 2) # Right eyebrow slanting down
    elif emotion == "surprised":
        # Round mouth (O shape)
        cv2.circle(img, (24, 32), 4, 220, 2)
        # Raised eyebrows
        cv2.ellipse(img, (16, 14), (5, 2), 0, 180, 360, 220, 2)
        cv2.ellipse(img, (32, 14), (5, 2), 0, 180, 360, 220, 2)
        
    # Apply slight blur to make features look organic/fuzzy
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Add random pixel noise
    gauss = np.random.normal(0, 5, img.shape).astype(np.float32)
    img_noisy = cv2.add(img.astype(np.float32), gauss)
    img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)
    
    return img_noisy

def generate_synthetic_dataset(num_train=200, num_test=50):
    """
    Generates a synthetic folder structure and grayscale images for the four emotions.
    """
    emotions = ["happy", "sad", "angry", "surprised"]
    splits = {
        "train": num_train,
        "test": num_test
    }
    
    print("Generating synthetic dataset...")
    for split, count in splits.items():
        split_dir = os.path.join(DATASET_DIR, split)
        for emotion in emotions:
            emotion_dir = os.path.join(split_dir, emotion)
            os.makedirs(emotion_dir, exist_ok=True)
            
            # Create a gitkeep file so folders are preserved
            with open(os.path.join(emotion_dir, ".gitkeep"), "w") as f:
                f.write("")
                
            print(f"Generating {count} images for split: {split}, emotion: {emotion}...")
            for i in range(count):
                img = draw_face_features(emotion)
                filename = os.path.join(emotion_dir, f"{emotion}_{i:04d}.png")
                cv2.imwrite(filename, img)
                
    print("Synthetic dataset generation complete!")

if __name__ == "__main__":
    download_haar_cascade()
    generate_synthetic_dataset()
