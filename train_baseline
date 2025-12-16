"""
Baseline MobileNetV2 Model for Waste Classification
End-to-end fine-tuning approach
Authors: Earl Jay G. Torayno, J Faye Champ Asaria
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import time
import os

def create_baseline_model(num_classes=6):
    """
    Create baseline MobileNetV2 model with custom classification head
    
    Args:
        num_classes: Number of waste categories (default: 6)
    
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2 without top layers
    base_model = MobileNetV2(
        include_top=False, 
        input_shape=(224, 224, 3), 
        weights="imagenet"
    )
    
    # Initially freeze base model
    base_model.trainable = False
    
    # Build complete model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    
    return model

def load_and_preprocess_data(data_dir, batch_size=32):
    """
    Load and preprocess TrashNet dataset
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for training (default: 32)
    
    Returns:
        Training and validation datasets
    """
    # Load training data
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(224, 224),
        batch_size=batch_size
    )
    
    # Load validation data
    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(224, 224),
        batch_size=batch_size
    )
    
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])
    
    normalization = layers.Rescaling(1./255)
    
    def augment_and_normalize(images, labels):
        images = data_augmentation(images)
        images = normalization(images)
        return images, labels
    
    def normalize_only(images, labels):
        images = normalization(images)
        return images, labels
    
    # Apply preprocessing
    train_ds = train_ds.map(augment_and_normalize)
    val_ds = val_ds.map(normalize_only)
    
    # Optimize performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    
    return train_ds, val_ds

def train_baseline_model(data_dir, epochs=30, batch_size=32):
    """
    Train the baseline MobileNetV2 model
    
    Args:
        data_dir: Path to dataset directory
        epochs: Number of training epochs (default: 30)
        batch_size: Batch size for training (default: 32)
    
    Returns:
        Trained model and training history
    """
    print("=== Baseline MobileNetV2 Training ===")
    print(f"Data directory: {data_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    
    # Load data
    train_ds, val_ds = load_and_preprocess_data(data_dir, batch_size)
    class_names = train_ds.class_names
    print(f"Classes: {class_names}")
    
    # Create model
    model = create_baseline_model(len(class_names))
    
    # Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Train model
    print("\nStarting training...")
    start_time = time.time()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Save model
    model_path = "baseline_mobilenetv2.keras"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model, history, training_time

def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Keras training history
    """
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('baseline_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Example usage
    data_directory = r"C:\Users\Station06\Torayno - IS\mini project\trashnet"
    
    # Train model
    model, history, training_time = train_baseline_model(data_directory)
    
    # Plot training history
    plot_training_history(history)
    
    print(f"\nFinal validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Training time: {training_time:.2f} seconds")
