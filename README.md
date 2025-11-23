
# 🌟 Wilms Tumor Detection Using Deep Learning 🌟

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-red)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.2+-green)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)](https://github.com)

---
Project overview

This project implements a YOLO-based pipeline to locate and classify Wilms tumor regions in MRI images. Key highlights from the paper:

Model: YOLO (YOLOv8 style architecture), backbone with CSP-like features; PANet + SPP neck. (See implementation diagrams in the paper.) 

project-paper

Dataset: Custom MRI dataset built from Radiopaedia examples — 20 patient cases (≈60 raw images) augmented to ~1000 images; train/val split ~80:20. 

project-paper

Typical training hyperparameters used in the study: Adam optimizer, learning rate 0.001, epochs=50 with early stopping, batch_size=16, image size 416x416. 

project-paper

Paper / dataset

Full project paper (PDF): /mnt/data/project-paper.pdf — use this local path for the paper. (It contains dataset description, architecture details, figures, training curves and the results table.) 

project-paper

Requirements

Create a Python venv and install dependencies:

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt


Example requirements.txt (adjust versions as required):

numpy
opencv-python
Pillow
tqdm
torch         # pick appropriate CUDA / CPU wheel
ultralytics   # if using Ultralytics YOLOv8 package
matplotlib
scikit-learn
albumentations


If you use a specific YOLO implementation (Ultralytics YOLOv8 or a custom repo), add/install that package accordingly.

Quick start — dataset & labels

Place images in data/images/ and YOLO-format label files in data/labels/ (same file name with .txt).

Example label format (one line per object):
class x_center y_center width height (normalized 0–1 relative to image size).

If you have bounding-box annotations in another format, use scripts/prepare_dataset.py to convert.

Training (example)

Minimal example (assumes Ultralytics yolo CLI or a train_yolo.py wrapper):

# using ultralytics package (example)
yolo task=detect mode=train model=yolov8n.pt \
  data=configs/wilms_dataset.yaml \
  epochs=50 imgsz=416 batch=16 lr0=0.001


Or with a train_yolo.py wrapper:

python scripts/train_yolo.py --data configs/wilms_dataset.yaml \
  --epochs 50 --batch 16 --img-size 416 --lr 0.001


Suggested hyperparameters (from the paper):

Optimizer: Adam

Learning rate: 0.001

Epochs: 50 (use early stopping on validation loss)

Batch size: 16

Image size: 416x416
These values reflect the setup described in the paper. 

project-paper

Inference (example)

Run detection on a folder of images:

python inference/detect.py --weights models/best.pt --source data/images/ --img-size 416 --conf-thres 0.25


inference/detect.py should:

Load model and weights

Run detection on the source images

Save annotated images to inference/results/

Optionally output a CSV/JSON with bounding boxes and scores

Evaluation & Results

The paper reports the following evaluation metrics (Table 1 / Results):
Accuracy: 97.40%
Precision: 96.55%
Recall: 98.26%
F1 score: 0.97
Dice coefficient: 0.97
(See the results table and training/validation loss curves in the paper.) 

project-paper

To evaluate on a test set, produce predictions and compute:

Precision, recall, F1 (per-class and overall)

IoU and Dice coefficient for segmentation-like overlap (for bounding boxes you can compute IoU-based Dice)

mAP @ IoU thresholds (e.g. 0.5)

Example evaluation snippet (very short):

from sklearn.metrics import precision_score, recall_score, f1_score

# load ground truth and predictions, compute per-image or aggregate metrics

Notes & tips

The dataset in the paper was small (20 patient cases → ~60 raw images) and augmented to ~1000 images. Use careful augmentation (rotation, flip, zoom) but avoid unrealistic transforms. 

project-paper

Use cross-validation (paper used 5-fold) to reduce overfitting risk and get robust estimates. 

project-paper

Use NMS (non-maximum suppression) in post-processing to remove duplicate boxes. Morphological refinements were applied in the study to tighten bounding-box edges. 

project-paper

How to reproduce figures in the paper

Training & validation loss curves: log losses (train/val) per epoch and plot using matplotlib.

The paper contains figures showing training/validation loss and sample detections — reference pages 5–6 for images and charts. 

project-paper

