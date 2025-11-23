🩺 Wilms Tumor Detection using Deep Learning (YOLOv8)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange?style=flat-square[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-success?style=flat-square&logo=ultralytics(https://github.com/ultralytics/ultralyticsge/License-MIT-yellow[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square

🚀 Project Overview
A deep learning pipeline for automated Wilms Tumor detection in pediatric kidney MRI scans. Leveraging YOLOv8 for real-time, high-accuracy tumor localization, this project aims to enhance diagnostic speed and precision in clinical settings.

Wilms Tumor is the most common renal cancer in children—early and accurate detection is vital for prognosis and treatment outcomes.

🎯 Problem Statement
Pediatric Wilms Tumor diagnosis is challenging and subject to human error.

Manual annotation of MRI scans is tedious, requiring experienced radiologists.

There is a pressing need for automated, robust, and fast tumor detection to assist medical professionals.

🌟 Solution
Use YOLOv8, a state-of-the-art object detection model, for fast and accurate tumor localization.

Preprocess and augment MRI datasets to increase training diversity.

Evaluate rigorously with multiple metrics (accuracy, precision, recall, F1, Dice).

Provide outputs for bounding box visualizations and clinical decision support.

📈 Results Highlights
Metric	Value
Accuracy	97.40%
Precision	96.55%
Recall	98.26%
F1-Score	0.97
Dice Coefficient	0.97
Training Dataset	~1000 images
Backbone Model	YOLOv8 (CSPDarknet, PANet, SPP)
✨ Features
End-to-End Pipeline: Data loading, augmentation, model training, evaluation, and visualization

Rich Augmentation: Rotation, flipping, zoom, intensity normalization

Modern Architecture: YOLOv8 for unified, fast diagnosis (single pass, multi-scale detection)

Bounding Box Outputs: For rapid review and manual correction

Cross-validation & Test Evaluation: Reliable performance estimates

User-friendly CLI: Simple command-line commands for training and inference

🛠️ Technology Stack
Area	Tool / Framework
Programming Language	Python 3.8+
DL Framework	TensorFlow, Keras
Detection Model	YOLOv8
Image Processing	OpenCV, PIL
Visualization	Matplotlib
Execution	Local, Google Colab
🏗️ Project Structure
text
Wilms-Tumor-Detection/
├── wilmstumordetection.py      # Complete pipeline script
├── project-paper.pdf           # Research, full methodology
├── README.md                   # You are here!
├── requirements.txt
│
├── data/
│   ├── images/                 # Raw MRI images
│   ├── labels/                 # YOLO-style bounding box annotations
│   └── augmented/              # Augmented data
│
├── models/
│   └── yolov8_weights.h5       # Trained weights
│
├── outputs/
│   ├── visualizations/         # Plots, example predictions
│   └── results.csv             # Metrics
⚡ Quick Start
Prerequisites
Python 3.8+

pip

Clone/download this repository

Install requirements

(Optional) Google Colab for rapid prototyping

Installation
bash
git clone https://github.com/YOUR_USERNAME/Wilms-Tumor-Detection.git
cd Wilms-Tumor-Detection
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
pip install -r requirements.txt
Usage
Data Augmentation

bash
python wilmstumordetection.py --augment --input_dir data/images --output_dir data/augmented
Training

bash
python wilmstumordetection.py --train --epochs 50 --batch_size 16 --lr 0.001
Evaluation

bash
python wilmstumordetection.py --eval --model_path models/yolov8_weights.h5 --test_dir data/test_images --test_labels data/test_labels
Visualization

bash
python wilmstumordetection.py --visualize --input_dir outputs/visualizations
🔬 Methodology
Dataset: MRI scans from 20 patients, with bounding boxes for Wilms tumors; augmented to ~1000 samples

Preprocessing: Normalization, resizing to 416x416, geometric and photometric augmentation

Model: YOLOv8 (CSPDarknet backbone, PANet and SPP neck, multi-scale head)

Training: Adam optimizer (lr=0.001), batch size 16, early stopping based on validation loss

Evaluation Metrics: Accuracy, precision, recall, F1-score, Dice coefficient

Post-processing: NMS for duplicate suppression, morphological box refinement

Validation: 5-fold cross-validation plus independent test set

📊 Sample Results
Class	Precision	Recall	F1	Dice
Wilms Tumor	96.5%	98.3%	0.97	0.97
Overall	96.4%	98.1%	0.97	0.97
📚 References and Documentation
Full details, ablation studies, and literature survey: project-paper.pdf

Source code: wilmstumordetection.py

🏆 Authors & Contributors
Venkatesh Kavididevi

Neeha Akhila Sri Kornepati

Deepak Reddy Chelladi

Shoaib Ali MD
Department of Information Technology, Vardhaman College of Engineering

🔗 Related Resources
YOLOv8

TensorFlow

Medical Imaging Datasets

📄 License
MIT – free for research, education, and non-commercial use.

🌟 If you found this useful, please ⭐ this repo and cite the project in your research!
Automating medical diagnostics for better outcomes, powered by deep learning and open data.

Related

Show a concise project overview and key features section

Provide a setup and installation quick start with commands

Add a usage examples section with code snippets

Include contribution guidelines and pull request template

Suggest badges and a clear license and citation block
\




