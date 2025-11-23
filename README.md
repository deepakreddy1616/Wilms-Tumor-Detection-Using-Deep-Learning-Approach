🎯 Wilms Tumor Detection using Deep Learning
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange?style=flat-square[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0+-(https://github.com/ultralytics/ultralyticshttps://img.shields.io/badge/License-MIT-yellow[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square

📋 Overview
This project presents a deep learning pipeline for automated detection of Wilms Tumor (pediatric kidney cancer) in MRI scans. Our approach integrates a customized YOLOv8 model, extensive image augmentation, and robust evaluation, aiming to support clinicians and radiologists in rapid and accurate diagnosis.

Motivation: Early detection is key to reducing relapse and improving pediatric outcomes. Machine learning enables consistent, reproducible, and precise detection, overcoming limitations of manual and conventional methods.

💡 Problem Statement
Manual diagnosis of Wilms Tumor is time-consuming, subjective, and error-prone.

Existing imaging techniques require expert review and may miss small or subtle tumors.

Need for a scalable, AI-powered system to identify tumors reliably and efficiently.

🧑‍🔬 Solution
Collect and augment a diverse dataset of pediatric MRI scans (Wilms Tumor and controls)

Preprocess images for uniformity and training efficiency

Deploy YOLOv8 architecture for fast, accurate object detection

Annotate and label tumor regions with bounding boxes

Evaluate performance using accuracy, precision, recall, F1-score, and Dice coefficient

🚀 Key Results
Metric	Value
Accuracy	97.40%
Precision	96.55%
Recall	98.26%
F1-Score	0.97
Dice coeff.	0.97
Training data	1000 images (augmented from 60 MRI scans/20 patients)
Model	YOLOv8 (custom config)
✨ Features
✅ End-to-end pipeline: Data pre-processing, augmentation, training, evaluation
✅ Extensive data augmentation (rotation, flipping, zooming)
✅ YOLOv8 architecture with CSPDarknet backbone, PANet and SPP neck, multi-scale detection head
✅ Bounding box labeling and visualization
✅ Cross-validation and independent test evaluation
✅ Robust post-processing (NMS, morphological box refinement)
✅ Performance plots (loss convergence, metrics over epochs)
✅ Modular, Pythonic codebase

🛠️ Technology Stack
Component	Technology
Programming Language	Python 3.8+
DL Framework	TensorFlow, Keras
Detection Model	YOLOv8
Image Processing	OpenCV, PIL
Data Augmentation	Custom routines
Environment	Google Colab/Jupyter
🏗️ Project Structure
text
Wilms-Tumor-Detection/
│
├── wilmstumordetection.py      # Main pipeline (preprocessing, train, eval, visualize)
├── project-paper.pdf           # Full research methodology & results
├── README.md                   # This file
└── requirements.txt            # Package dependencies

├── data/
│   ├── images/                 # Source MRI scans
│   ├── labels/                 # Bounding box annotations
│   └── augmented/              # Augmented image data
│
├── models/
│   └── yolov8_weights.h5       # Trained model weights
├── outputs/
│   ├── visualizations/         # Detection and metric plots
│   └── results.csv             # Evaluation metrics
📦 Installation
Prerequisites
Python 3.8 or above

pip

Git

Steps
bash
git clone https://github.com/YOUR_USERNAME/Wilms-Tumor-Detection.git
cd Wilms-Tumor-Detection

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
⚡ Usage
1. Data Augmentation
bash
python wilmstumordetection.py --augment \
  --input_dir data/images \
  --label_dir data/labels \
  --output_dir data/augmented
2. Training
bash
python wilmstumordetection.py --train \
  --epochs 50 \
  --batch_size 16 \
  --lr 0.001
3. Evaluation
bash
python wilmstumordetection.py --evaluate \
  --model_path models/yolov8_weights.h5 \
  --test_dir data/test_images \
  --test_labels data/test_labels
4. Visualization
bash
python wilmstumordetection.py --visualize \
  --input_dir outputs/visualizations
🧠 Methodology
Dataset Construction: Axial portal venous phase MRI scans from 20 patients, augmented to ~1000 images

Preprocessing: Normalization, resizing (416x416 px), geometric and photometric augmentation

Model Architecture: YOLOv8 (CSPDarknet backbone, PANet/SPP neck, multi-scale output head)

Labeling: Manual bounding box annotation per scan (txt/Yolo format)

Training: Adam optimizer, composite loss for classification and localization, early stopping

Validation: 5-fold cross-validation plus test set

Post-processing: NMS and morphological refinements for bounding boxes

Evaluation: On independent test set—reporting accuracy, precision, recall, F1, Dice

📊 Training and Test Performance
Metric	Training	Test
Accuracy	97.4%	>96%
Precision	96.6%	>95%
Recall	98.3%	>97%
F1-Score	0.97	>0.96
Dice coefficient	0.97	>0.96
Training curves and sample result visualizations available in outputs/visualizations/.

🔬 Research Context
This model leverages and advances current best practices from automated tumor classification studies (see Literature Review in project-paper.pdf), demonstrating superior accuracy and reliability in pediatric MRI interpretation.

⚙️ Configuration
YOLOv8 settings, hyperparameters, and training routines are fully customizable via command-line flags. See wilmstumordetection.py --help for all options.

📚 Documentation
Full research paper, dataset details, methodology, and references in project-paper.pdf

Annotated, readable code in wilmstumordetection.py

For questions, contact the authors

🏆 Authors & Collaborators
Venkatesh Kavididevi

Neeha Akhila Sri Kornepati

Deepak Reddy Chelladi

Shoaib Ali MD

Department of Information Technology, Vardhaman College of Engineering

🔗 References
Key references from academic literature included in the project paper—benchmarking deep learning in Wilms Tumor detection and segmentation.

📄 License
MIT License – free for academic, medical, and non-commercial use.
