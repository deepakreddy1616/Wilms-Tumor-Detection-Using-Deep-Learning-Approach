
# 🌟 Wilms Tumor Detection Using Deep Learning 🌟

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-red)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.2+-green)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)](https://github.com)

---
## 📌 Project Overview

Wilms tumor is the most frequent kidney cancer in children. Early detection improves survival. 
It  is an AI-powered project that automates the identification of pediatric kidney tumors in MRI scans using YOLOv8 deep learning.
It leverages advanced data augmentation and state-of-the-art object detection to localize tumors quickly and accurately.

---

## ✨ Key Features

| Feature                 | Description                                                           |
|-------------------------|-----------------------------------------------------------------------|
| YOLOv8 Object Detection | Detects and bounds Wilms tumor in medical images (MRI)                |
| High Accuracy           | 97.4% accuracy, F1=0.97, Dice=0.97                                   |
| Augmentation            | Resizing, normalization, strong augmentation prevents overfitting      |
| Fast Inference          | Real-time prediction in clinical environments                         |
| Flexible/Extensible     | Modular code, easy to adapt for other medical imaging tasks           |

---

## 🛠️ Technology Stack

- **Python 3.10+**
- **PyTorch**
- **Ultralytics YOLOv8**
- **Albumentations**
- **OpenCV**
- **Google Colab** (for fast prototyping)
  
---
## Architecture

- Inputs: MRI (Radiopaedia), YOLO labels
- Model: YOLOv8 (CSPDarknet, PANet, SPP)
- Outputs: Tumor bbox, metrics, plots
---
Raw MRI Images
      |
Preprocessing & Augmentation (Albumentations, OpenCV)
      |
YOLOv8 Model Training (Ultralytics)
      |
Tumor Localization & Output (bounding boxes, metrics)
      |
Clinical Reporting (plots, PDF summaries)

---



## ⚡ Performance Summary

| Metric       | Value    |
|--------------|----------|
| Accuracy     | 97.40%   |
| Sensitivity  | 98.26%   |
| Precision    | 96.55%   |
| F1 Score     | 0.97     |
| Dice Coef.   | 0.97     |

---
## 🖼️ Example Output

![Tumor Detection Sample](docs/example_detection.png	Value
Accuracy	97.4%
F1 Score	0.97
Dice Coef.	0.97

---

---
## 🚀 Usage
Train model:
python wilmstumordetection.py --train --epochs 50

Run inference:
python wilmstumordetection.py --detect --input images/test_mri.jpg

Help:
python wilmstumordetection.py --help

---
## Docs and References
- [YOLOv8 Docs](https://docs.ultralytics.com/)
- Radiopaedia.org

---



*Optimizing pediatric cancer care with code*
