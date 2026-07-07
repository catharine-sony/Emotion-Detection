"""
Training script for the Emotion Detection CNN model.
Loads preprocessed training and validation images, compiles the CNN model,
performs training, plots accuracy/loss metrics, and saves the trained model.
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Import utilities to auto-download cascades or generate datasets if missing
from utils import download_haar_cascade, generate_synthetic_dataset

# Set configurations
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 20
CLASSES = ["angry", "happy", "sad", "surprised"]
NUM_CLASSES = len(CLASSES)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

def verify_dataset():
    """
    Checks if train and test folders have images.
    If directories are empty or missing, calls utils to generate synthetic datasets.
    """
    download_haar_cascade()
    
    dataset_valid = True
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        dataset_valid = False
    else:
        # Check if subfolders contain files
        for split in ['train', 'test']:
            split_path = os.path.join(DATASET_DIR, split)
            for emotion in CLASSES:
                emotion_path = os.path.join(split_path, emotion)
                if not os.path.exists(emotion_path) or len(os.listdir(emotion_path)) <= 1:
                    # Only has .gitkeep or is empty
                    dataset_valid = False
                    break
            if not dataset_valid:
                break
                
    if not dataset_valid:
        print("Dataset directory is empty or missing. Generating synthetic data...")
        generate_synthetic_dataset(num_train=300, num_test=75)
    else:
        print("Existing dataset directory verified successfully.")

def build_model(input_shape=(48, 48, 1), num_classes=4):
    """
    Builds a robust, production-quality CNN model for grayscale facial expression classification.
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Fully Connected Block
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_training_results(history, output_path):
    """
    Plots training & validation accuracy and loss, saving the figure.
    """
    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='#3498db', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='#e74c3c', linewidth=2)
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='#3498db', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', color='#e74c3c', linewidth=2)
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Training history plot saved to {output_path}")
    plt.close()

def main():
    # Make sure we have data
    verify_dataset()
    
    # 1. Set up data pipelines (Grayscale, normalization rescale=1./255)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    print("\nLoading training images...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=True
    )
    
    print("\nLoading test images...")
    validation_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False
    )
    
    # 2. Build the model
    print("\nInitializing CNN model...")
    model = build_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1), num_classes=NUM_CLASSES)
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 3. Callbacks for model checkpointing and optimization
    model_file_name = "emotion_model.keras"
    model_save_path = os.path.join(MODELS_DIR, model_file_name)
    
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
    
    # 4. Train the model
    print(f"\nStarting model training for {EPOCHS} epochs...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    # Save the final model (as a fallback or in case early stopping is not triggered)
    # The checkpoint callback already saves the best validation accuracy model
    print(f"\nTraining completed. Saving final model to {model_save_path}...")
    model.save(model_save_path)
    
    # 5. Evaluate the model on test set
    test_loss, test_acc = model.evaluate(validation_generator)
    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # 6. Plot and save accuracy / loss graphs
    plot_path = os.path.join(MODELS_DIR, 'training_history.png')
    plot_training_results(history, plot_path)

if __name__ == "__main__":
    main()
