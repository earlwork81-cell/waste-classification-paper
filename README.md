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
