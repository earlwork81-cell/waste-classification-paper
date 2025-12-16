
### hybrid_model/README.md:
```markdown
# Hybrid MobileNetV2 + SVM Model

This directory contains the hybrid implementation using frozen MobileNetV2 for feature extraction and SVM for classification.

## Files
- `train_hybrid.py` - Main training script
- `evaluate_hybrid.py` - Model evaluation utilities

## Usage
```python
from train_hybrid import train_hybrid_model

# Train model
feature_extractor, svm_model, results, training_time = train_hybrid_model(
    data_dir="path/to/trashnet",
    kernel='rbf',
    C=5,
    gamma='scale'
)
