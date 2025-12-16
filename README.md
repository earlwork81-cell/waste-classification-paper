# Baseline MobileNetV2 Model

This directory contains the baseline MobileNetV2 implementation with end-to-end fine-tuning.

## Files
- `train_baseline.py` - Main training script
- `evaluate_baseline.py` - Model evaluation utilities

## Usage
```python
from train_baseline import train_baseline_model

# Train model
model, history, training_time = train_baseline_model(
    data_dir="path/to/trashnet",
    epochs=30,
    batch_size=32
)

Model Architecture
Base: MobileNetV2 (ImageNet pre-trained)
Classification Head: Dense(128) → Dropout(0.3) → Dense(6)
Input Size: 224×224×3
Output: 6 classes (softmax)
Training Details
Optimizer: Adam
Loss: Sparse Categorical Crossentropy
Epochs: 30
Batch Size: 32
Data Augmentation: Random flip, rotation, zoom
