
# 🌟 Wilms Tumor Detection Using Deep Learning 🌟

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-red)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.2+-green)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)](https://github.com)

---
## 📌 Project Overview

Wilms tumor is the most frequent kidney cancer in children. 
Early detection improves survival. 
This project applies deep learning (YOLOv8) for precise object detection in pediatric MRIs, automating tumor localization for clinical decision support.

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

## How To Use

**Train:**
`python wilmstumordetection.py --train --epochs 50`

**Evaluate or Predict:**
`python wilmstumordetection.py --detect --input images/your_mri.jpg`

**Visual Results:**
`python -m scripts.visualize`


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
## ⚙️ Installation & Setup
bash
git clone https://github.com/YOUR_USERNAME/Wilms-Tumor-Detection-Using-Deep-Learning.git
cd Wilms-Tumor-Detection-Using-Deep-Learning
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
pip install -r requirements.txt

---
## 🚀 Usage
Train model:
python wilmstumordetection.py --train --epochs 50

Run inference:
python wilmstumordetection.py --detect --input images/test_mri.jpg

Help:
python wilmstumordetection.py --help

---
## 🧾 API Documentation
python
from wilmstumordetection import WilmsTumorDetector
detector = WilmsTumorDetector(model_path='weights/best.pt')
output = detector.predict('images/sample_mri.jpg')
print(output)  # {'boxes':[], 'classes':[], 'scores':[]}
Main API:

train() - Model training

predict(image_path) - Prediction from file

visualize(image_path) - Annotated output

----
## 🧪 Testing
Run: pytest tests/

Coverage: > 85%
---

## Docs and References
- [YOLOv8 Docs](https://docs.ultralytics.com/)
- Radiopaedia.org

---



*Optimizing pediatric cancer care with code*
