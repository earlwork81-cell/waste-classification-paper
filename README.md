# Waste Classification Research Paper

## Authors
- **Earl Jay G. Torayno** - South Philippine Adventist College
- **J Faye Champ Asaria** - South Philippine Adventist College

## Paper Title
**Comparative Study of MobileNetV2 and Hybrid MobileNetV2 with SVM Classifier for Efficient Waste Classification in Low-Resource Environments**

## Conference
Philippine Computing Science Congress (PCSC) 2024  
Laguna, Philippines  
May 2024

## Abstract
This study presents a comparative investigation between standard MobileNetV2 and a proposed Hybrid MobileNetV2 integrated with Support Vector Machine (SVM) classifier for waste classification. Using the TrashNet dataset with 2,527 labeled images across six waste categories, we demonstrate that the hybrid approach achieves 87.52% accuracy compared to 82.38% for the baseline, while being 20× faster and 56% smaller.

## Quick Results Summary

| Metric | Baseline | Hybrid | Improvement |
|--------|----------|---------|-------------|
| Accuracy (%) | 82.38 | 87.52 | +5.14% |
| Inference Time (ms) | 28.0 | 1.39 | 20.1× faster |
| Model Size (MB) | 9.24 | 4.1 | 56% smaller |
| Training Time (s) | 1623.56 | 1.93 | 840× faster |

## Repository Structure
- `latex/` - LaTeX source files
  - `PCSC2024-sigconf.tex` - Main document
  - `samplebody-conf.tex` - Paper content
  - `acmart.cls` - ACM document class
  - `references.bib` - Bibliography

## Key Findings
✅ **Superior Accuracy**: 87.52% vs 82.38% (+5.14%)  
✅ **Faster Inference**: 20.1× speed improvement  
✅ **Smaller Model**: 56% size reduction  
✅ **Efficient Training**: 840× faster training  

## Per-Class Performance (Hybrid Model)
| Category | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Cardboard | 0.94 | 0.87 | 0.90 |
| Glass | 0.90 | 0.93 | 0.92 |
| Metal | 0.85 | 0.92 | 0.89 |
| Paper | 0.89 | 0.94 | 0.91 |
| Plastic | 0.82 | 0.81 | 0.81 |
| Trash | 0.82 | 0.53 | 0.64 |

## Methodology
1. **Baseline**: End-to-end MobileNetV2 fine-tuning
2. **Hybrid**: Frozen MobileNetV2 + RBF SVM classifier
3. **Dataset**: TrashNet (2,527 images, 6 categories)
4. **Evaluation**: Accuracy, F1-score, inference time, model size

## Compilation
This paper uses the ACM conference template. To compile:
1. Upload all files to Overleaf
2. Set main document to `PCSC2024-sigconf.tex`
3. Compile with pdflatex + bibtex

## Contact
- Earl Jay G. Torayno: earljay.torayno@spac.edu.ph
- J Faye Champ Asaria: fayechamp.asaria@spac.edu.ph

---
*Repository created for PCSC 2024 submission*
