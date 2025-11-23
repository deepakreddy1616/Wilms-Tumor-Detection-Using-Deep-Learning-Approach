
# 🌟 Wilms Tumor Detection Using Deep Learning (YOLOv8) 🌟

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-red)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.2+-green)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)](https://github.com)

---

## 📑 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [🛠️ Technology Stack](#️technology-stack)
- [🔌 APIs & Data Sources](#️apis--data-sources)
- [🔧 Development & Testing](#development--testing)
- [🏗️ System Architecture](#system-architecture)
- [Installation & Setup](#installation--setup)
    - [Step-by-Step Installation](#step-by-step-installation)
    - [Quick Start](#quick-start)
    - [Usage & Examples](#usage--examples)
    - [Using as Python Module](#using-as-python-module)
    - [Advanced Configuration](#advanced-configuration)
- [Example Output](#example-output)
- [⚡ Performance Results](#-performance-results)
- [📊 Quantitative Metrics](#quantitative-metrics)
- [📄 PI Documentation](#pi-documentation)
- [🧪 Testing](#testing)
- [📚 Research References](#research-references)
- [✍️ Authors](#authors)

---

## 🔬 Project Overview

Wilms tumor (Nephroblastoma) is the most common kidney cancer in children. Early detection is crucial for effective treatment. This project leverages deep learning, specifically the YOLOv8 object detection architecture, to automate and optimize tumor identification in MRI scans.

---

## ✨ Key Features

- **YOLOv8-based Detection:** Real-time, high-accuracy object detection.
- **End-to-End Pipeline:** Data preprocessing, augmentation, training, evaluation.
- **Comprehensive Metrics:** mAP, Precision, Recall, F1, Dice coefficient.
- **Robust Preprocessing:** Normalization, resizing, augmentation (rotation, flipping, zooming) for generalizability.
- **Flexible API:** Can be used as a CLI script or Python module.
- **Rich Visualization:** Training curves, confusion matrix, detection overlays.
- **Modular Design:** Easily extensible for further research.

---

## 🛠️ Technology Stack

| Layer             | Technology      | Purpose               |
|-------------------|----------------|-----------------------|
| Language          | Python 3.10+    | Core development      |
| Deep Learning     | PyTorch         | Model training        |
| Object Detection  | Ultralytics YOLOv8 | Tumor localization |
| Augmentation      | Albumentations  | Data enrichment       |
| Processing        | OpenCV, PIL     | Image manipulation    |
| Environment       | Google Colab    | GPU-enabled training  |
| Visualization     | Matplotlib      | Plots & charts        |

---

## 🔌 APIs & Data Sources

- **Radiopedia**: Source for sample pediatric MRI data.
- **Internal Dataset**: Augmented and annotated Wilms tumor and non-tumor MRI images (not publicly available).
- **YOLOv8 API**: For model training and inference.

---

## 🔧 Development & Testing

- Developed and iterated using Git & GitHub.
- Unit and integration testing for model outputs and pipeline modules.
- Continuous Integration with GitHub Actions (optional).
- Pre-trained weights available for initial evaluation.

---

## 🏗️ System Architecture

graph TD
A[Raw MRI Images] --> B[Preprocessing & Augmentation]
B --> C[YOLOv8 Model]
C --> D[Evaluation Metrics]
C --> E[Detection & Visualization]

text

- **Input:** MRI scan images.
- **Process:** Preprocessing (resize, normalize, augment) → YOLOv8 detection → Postprocessing.
- **Output:** Detected tumor bounding boxes, scored metrics.

---

## Installation & Setup

### Step-by-Step Installation

Clone the repo
git clone https://github.com/YOUR_USERNAME/Wilms-Tumor-Detection-Using-Deep-Learning-Approach.git
cd Wilms-Tumor-Detection-Using-Deep-Learning-Approach

Create Python virtual environment
python -m venv venv

Activate (Windows)
venv\Scripts\activate

Activate (Mac/Linux)
source venv/bin/activate

Install dependencies
pip install -r requirements.txt

text

### Quick Start

Data Augmentation
python wilmstumordetection.py --augment --input_images data/images --input_labels data/labels --output_dir data/augmented

Train YOLOv8 Model
python wilmstumordetection.py --train --epochs 50 --batch_size 16

Inference Example
python wilmstumordetection.py --detect --input images/test_image.jpg

text

### Usage & Examples

import wilmstumordetection

Run preprocessing and training programmatically
wilmstumordetection.preprocess(...)
wilmstumordetection.train_model(epochs=50)
results = wilmstumordetection.infer(img_path='images/sample.jpg')
print(results)

text

### Using as Python Module

from wilmstumordetection import WilmsTumorDetector

detector = WilmsTumorDetector(model_path='best.pt')
results = detector.predict('images/test_mri.jpg')

text

### Advanced Configuration

See config arguments in:
python wilmstumordetection.py --help

text

---

## Example Output

![Detection Example](docs/example_detection.png)

| Metric         | Value   |
|----------------|---------|
| Accuracy       | 97.4%   |
| F1 Score       | 0.97    |
| Dice Coef.     | 0.97    |

---

## ⚡ Performance Results

Project results are summarized below. See detailed interactive dashboard [here](https://github.com/deepakreddy1616/Real-time-logistics-routing-during-emergency-using-metaheuristic-algorithms#-performance-results).

| Metric         | Value   |
|----------------|---------|
| Accuracy       | 97.4%   |
| Recall         | 98.26%  |
| Precision      | 96.55%  |

---

## 📊 Quantitative Metrics

- **Training Set:** 1000 images (augmented)
- **Validation Set:** 200 images
- **Test Set:** 200 images
- **Epochs:** 50
- **Optimizer:** Adam
- **Batch Size:** 16

---

## 📄 PI Documentation

Detailed methodology, dataset, and results are available in [`project-paper.pdf`](project-paper.pdf).

---

## 🧪 Testing

- Unit tests for data preprocessing and metric calculations.
- Evaluation scripts included in the notebook and main script.
- See `tests/` folder for sample test cases (add as needed).

---

## 📚 Research References

- Venkatesh Kavididevi, Necha Akhila Sri Kornepati, Deepak Reddy Chelladi, Shoaib Ali MD, Vardhaman College of Engineering, Hyderabad, India.
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Radiopaedia Medical Image Database](https://radiopaedia.org)

---

## ✍️ Authors

- **Venkatesh Kavididevi**
- **Necha Akhila Sri Kornepati**
- **Deepak Reddy Chelladi**
- **Shoaib Ali MD**

Department of Information Technology, Vardhaman College of Engineering, Hyderabad, India

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

*Built with ❤️ using Python, YOLOv8, and PyTorch*


