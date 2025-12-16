# Waste Classification Code Repository

## Authors
- **Earl Jay G. Torayno** - South Philippine Adventist College
- **J Faye Champ Asaria** - South Philippine Adventist College

## Project Description
Implementation of two waste classification models:
1. **Baseline MobileNetV2** - End-to-end fine-tuning approach
2. **Hybrid MobileNetV2+SVM** - Feature extraction with SVM classification

## Dataset
- **TrashNet Dataset**: 2,527 labeled images across 6 categories
- **Categories**: cardboard, glass, metal, paper, plastic, trash
- **Split**: 80% training, 20% validation

## Key Results
| Model | Accuracy | Inference Time | Model Size |
|-------|----------|----------------|------------|
| Baseline MobileNetV2 | 82.38% | 28.0 ms/img | 9.24 MB |
| Hybrid MobileNetV2+SVM | 87.52% | 1.39 ms/img | 4.1 MB |

## Repository Structure

### Code Organization

waste-classification-code/
├── baseline_model/          # Baseline MobileNetV2 implementation
├── hybrid_model/            # Hybrid MobileNetV2+SVM implementation
├── data_preprocessing/      # Data loading and preprocessing utilities
├── utils/                   # Helper functions and utilities
├── requirements.txt         # Python dependencies
└── README.md               # This file


### Quick Start
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   Train baseline model:
bash
Copy
cd baseline_model
python train_baseline.py
Train hybrid model:
bash
Copy
cd hybrid_model
python train_hybrid.py
Model Details
Baseline MobileNetV2
Transfer learning with ImageNet weights
End-to-end fine-tuning for 30 epochs
Achieves 82.38% validation accuracy
Hybrid MobileNetV2+SVM
Frozen MobileNetV2 as feature extractor
RBF kernel SVM for classification
Achieves 87.52% validation accuracy
20× faster inference, 56% smaller model
Requirements
Python 3.8+
TensorFlow 2.10
scikit-learn 1.1.1
NumPy, Pandas, Matplotlib, Seaborn
Installation
bash
Copy
# Clone repository
git clone https://github.com/YOUR_USERNAME/waste-classification-code.git
cd waste-classification-code

# Install dependencies
pip install -r requirements.txt
Usage Examples
Training Models
Python
Copy
# Train baseline model
from baseline_model.train_baseline import train_baseline_model
model = train_baseline_model(data_path="path/to/trashnet")

# Train hybrid model  
from hybrid_model.train_hybrid import train_hybrid_model
model = train_hybrid_model(data_path="path/to/trashnet")
Evaluation
Python
Copy
# Evaluate models
from baseline_model.evaluate_baseline import evaluate_model
results = evaluate_model(model, test_data)

from hybrid_model.evaluate_hybrid import evaluate_model
results = evaluate_model(model, test_data)
Performance Metrics
Accuracy: Overall classification accuracy
F1-Score: Harmonic mean of precision and recall
Inference Time: Average prediction time per image
Model Size: Memory footprint in MB
Contact
Earl Jay G. Torayno: earljay.torayno@spac.edu.ph
J Faye Champ Asaria: fayechamp.asaria@spac.edu.ph
License
This project is licensed under the MIT License - see LICENSE file for details.
