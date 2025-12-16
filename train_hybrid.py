"""
Hybrid MobileNetV2 + SVM Model for Waste Classification
Feature extraction with SVM classification
Authors: Earl Jay G. Torayno, J Faye Champ Asaria
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib

def load_and_preprocess_data(data_dir, batch_size=32):
    """
    Load and preprocess TrashNet dataset for hybrid model
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for processing (default: 32)
    
    Returns:
        Training and test datasets
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
    
    # Load test data
    test_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(224, 224),
        batch_size=batch_size
    )
    
    # Data augmentation (only for training)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])
    
    def augment_and_preprocess(images, labels):
        images = data_augmentation(images)
        images = preprocess_input(images)
        return images, labels
    
    def preprocess_only(images, labels):
        images = preprocess_input(images)
        return images, labels
    
    # Apply preprocessing
    train_ds = train_ds.map(augment_and_preprocess)
    test_ds = test_ds.map(preprocess_only)
    
    # Optimize performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)
    
    return train_ds, test_ds

def create_feature_extractor():
    """
    Create frozen MobileNetV2 feature extractor
    
    Returns:
        Feature extractor model
    """
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        include_top=False,
        input_shape=(224, 224, 3),
        weights="imagenet"
    )
    
    # Freeze all layers
    base_model.trainable = False
    
    # Create feature extractor
    feature_extractor = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
    ])
    
    return feature_extractor

def extract_features(feature_extractor, dataset):
    """
    Extract features from dataset using the feature extractor
    
    Args:
        feature_extractor: Frozen MobileNetV2 model
        dataset: TensorFlow dataset
    
    Returns:
        Features and labels as numpy arrays
    """
    all_features = []
    all_labels = []
    
    print("Extracting features...")
    for images, labels in dataset:
        features = feature_extractor(images)
        all_features.append(features.numpy())
        all_labels.append(labels.numpy())
    
    return np.concatenate(all_features), np.concatenate(all_labels)

def train_svm_classifier(X_train, y_train, kernel='rbf', C=5, gamma='scale'):
    """
    Train SVM classifier on extracted features
    
    Args:
        X_train: Training features
        y_train: Training labels
        kernel: SVM kernel (default: 'rbf')
        C: Regularization parameter (default: 5)
        gamma: Kernel coefficient (default: 'scale')
    
    Returns:
        Trained SVM model and training time
    """
    print(f"Training SVM classifier...")
    print(f"Kernel: {kernel}, C: {C}, gamma: {gamma}")
    print(f"Training samples: {len(X_train)}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    
    # Create SVM classifier
    svm_model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        random_state=42
    )
    
    # Train SVM
    start_time = time.time()
    svm_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"SVM training completed in {training_time:.2f} seconds")
    print(f"Number of support vectors: {len(svm_model.support_)}")
    
    return svm_model, training_time

def evaluate_model(feature_extractor, svm_model, X_test, y_test, class_names):
    """
    Evaluate the hybrid model on test data
    
    Args:
        feature_extractor: Frozen MobileNetV2 model
        svm_model: Trained SVM classifier
        X_test: Test features
        y_test: Test labels
        class_names: List of class names
    
    Returns:
        Evaluation metrics
    """
    print("\n=== Model Evaluation ===")
    
    # Measure inference time
    start_time = time.time()
    y_pred = svm_model.predict(X_test)
    inference_time = (time.time() - start_time) / len(y_test) * 1000
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test accuracy: {accuracy*100:.2f}%")
    print(f"Inference time: {inference_time:.2f} ms/image")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Hybrid MobileNetV2 + SVM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig('hybrid_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'inference_time': inference_time,
        'classification_report': classification_report(y_test, y_pred, target_names=class_names, output_dict=True),
        'confusion_matrix': cm
    }

def train_hybrid_model(data_dir, kernel='rbf', C=5, gamma='scale'):
    """
    Complete training pipeline for hybrid model
    
    Args:
        data_dir: Path to dataset directory
        kernel: SVM kernel (default: 'rbf')
        C: Regularization parameter (default: 5)
        gamma: Kernel coefficient (default: 'scale')
    
    Returns:
        Trained models and evaluation results
    """
    print("=== Hybrid MobileNetV2 + SVM Training ===")
    print(f"Data directory: {data_dir}")
    print(f"SVM Kernel: {kernel}, C: {C}, gamma: {gamma}")
    
    # Load data
    train_ds, test_ds = load_and_preprocess_data(data_dir)
    class_names = train_ds.class_names
    print(f"Classes: {class_names}")
    
    # Create feature extractor
    feature_extractor = create_feature_extractor()
    print("\nFeature Extractor Summary:")
    feature_extractor.summary()
    
    # Extract features
    print("\nExtracting training features...")
    X_train, y_train = extract_features(feature_extractor, train_ds)
    
    print("Extracting test features...")
    X_test, y_test = extract_features(feature_extractor, test_ds)
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    # Train SVM
    svm_model, training_time = train_svm_classifier(X_train, y_train, kernel, C, gamma)
    
    # Evaluate model
    results = evaluate_model(feature_extractor, svm_model, X_test, y_test, class_names)
    
    # Save models
    feature_extractor_path = "hybrid_feature_extractor.keras"
    svm_model_path = "hybrid_svm_classifier.joblib"
    
    feature_extractor.save(feature_extractor_path)
    joblib.dump(svm_model, svm_model_path)
    
    print(f"\nModels saved:")
    print(f"Feature extractor: {feature_extractor_path}")
    print(f"SVM classifier: {svm_model_path}")
    
    return feature_extractor, svm_model, results, training_time

if __name__ == "__main__":
    # Example usage
    data_directory = r"C:\Users\Station06\Torayno - IS\mini project\trashnet"
    
    # Train hybrid model
    feature_extractor, svm_model, results, training_time = train_hybrid_model(data_directory)
    
    print(f"\n=== Final Results ===")
    print(f"Test accuracy: {results['accuracy']*100:.2f}%")
    print(f"Inference time: {results['inference_time']:.2f} ms/image")
    print(f"Training time: {training_time:.2f} seconds")
