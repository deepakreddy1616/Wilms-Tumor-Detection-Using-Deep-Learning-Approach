🧠 Wilms Tumor Detection using Deep Learning (YOLO)

A deep learning–based medical imaging project to detect Wilms tumor in pediatric MRI scans using a YOLO architecture.
This repository contains code, documentation, trained model files, and the full project paper.

📄 Full Project Paper: project-paper.pdf (included in repo)

⭐ Project Highlights

🚀 YOLO-based object detection model tailored for MRI tumor detection

🎯 Achieves 97.4% accuracy, 96.55% precision, 98.26% recall, F1 = 0.97

🧪 Dataset of MRI scans, preprocessed & augmented to improve robustness

🧩 Includes training scripts, inference code, model weights, and evaluation

📚 Built as part of an academic deep learning research project

📘 Table of Contents

Overview

Dataset

Results

Tech Stack

Repository Structure

Setup & Installation

Training

Inference

Contact

🔍 Overview

Wilms tumor is a kidney cancer commonly found in children.
This project builds an end-to-end deep learning pipeline to automatically detect tumor regions on MRI scans.

We use a YOLO architecture with:

CSP-type backbone

PANet + SPP neck

YOLO detection head

Adam optimizer

Image size: 416×416

Epochs: 50

Batch size: 16

🗂 Dataset

MRI images collected from open-source radiology resources

20 patient cases → ~60 raw images

Data augmented → 1000+ images

Labeled using YOLO bounding-box format

Train/Val split: 80/20

🧪 Results
Metric	Score
Accuracy	97.40%
Precision	96.55%
Recall	98.26%
F1 Score	0.97
Dice Coefficient	0.97

The model reliably detects Wilms tumor regions with strong performance across all metrics.

🛠 Tech Stack

Python

PyTorch

YOLO architecture (Ultralytics-based or custom implementation)

OpenCV

Albumentations

NumPy / Matplotlib

Scikit-learn

📁 Repository Structure
.
├── README.md
├── project-paper.pdf
├── data/
│   ├── images/
│   └── labels/
├── scripts/
│   ├── prepare_dataset.py
│   ├── augment.py
│   └── train_yolo.py
├── inference/
│   └── detect.py
├── models/
│   └── best.pt
├── configs/
│   └── dataset.yaml
└── requirements.txt

⚙️ Setup & Installation
git clone <your-repo-link>
cd <repo-folder>

python -m venv venv
source venv/bin/activate        # Windows → venv\Scripts\activate

pip install -r requirements.txt


Example requirements.txt:

torch
opencv-python
ultralytics
numpy
matplotlib
albumentations
scikit-learn
Pillow
tqdm

🏋️ Training

Using Ultralytics YOLO:

yolo task=detect mode=train model=yolov8n.pt \
  data=configs/dataset.yaml \
  epochs=50 batch=16 imgsz=416 lr0=0.001


OR using your custom script:

python scripts/train_yolo.py --epochs 50 --batch 16 --img-size 416

🔮 Inference

Run detection on sample images:

python inference/detect.py --weights models/best.pt --source data/images/


Results will be saved in:

inference/results/

👨‍⚕️ Applications

Early tumor screening

Radiology workflow assistance

Decision support systems

Medical AI research

📬 Contact

📧 chelladideepakreddy@gmail.com

💼 LinkedIn: Add your link here

📜 License

This project is licensed under the MIT License.
Feel free to use, modify, and contribute.
