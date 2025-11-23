# 🌟 Wilms Tumor Detection using Deep Learning (YOLOv8) 🌟

## 🔬 Project Overview

[cite_start]Wilms tumor (Nephroblastoma) is the most common pediatric kidney cancer, and its swift, accurate diagnosis is crucial for effective treatment and improved patient outcomes[cite: 349, 351].

This project presents an **innovative deep learning-based system** for the early and automated detection of Wilms tumors in MRI scans. [cite_start]By leveraging the power of the **You Only Look Once (YOLOv8)** architecture, our model provides a fast, accurate, and automated diagnostic tool designed to assist radiologists and significantly enhance clinical efficiency[cite: 345, 566, 571, 572].

---

## ✨ Key Features

* [cite_start]**State-of-the-Art Architecture:** Utilizes the high-efficiency and accuracy of the **YOLOv8** object detection model[cite: 463, 500].
* [cite_start]**High Diagnostic Accuracy:** Achieved an outstanding accuracy of **97.40%** and a Dice coefficient of **0.97** on the evaluation dataset[cite: 550, 552].
* [cite_start]**Real-Time Potential:** YOLO's speed and unified architecture make it highly suitable for real-time clinical applications where timely diagnosis is critical[cite: 533, 534, 571].
* [cite_start]**Robust Model Training:** Implements comprehensive preprocessing steps including **Normalization, Resizing (416x416), and extensive Augmentation** (rotation, flipping, zooming) to prevent overfitting and ensure generalizability[cite: 458, 459, 460, 569].

---

## 🚀 Performance Metrics

[cite_start]Our model was rigorously evaluated on an augmented dataset of nearly 1000 MRI images, demonstrating strong performance across key metrics[cite: 448, 547, 550]:

| Metric | Value |
| :--- | :--- |
| **Accuracy** | [cite_start]**97.40%** [cite: 550] |
| Recall (Sensitivity) | [cite_start]98.26% [cite: 552] |
| Precision | [cite_start]96.55% [cite: 551] |
| **F1 Score** | [cite_start]**0.97** [cite: 552] |
| **Dice Coefficient** | [cite_start]**0.97** [cite: 552] |

---

## ⚙️ Methodology

[cite_start]The core of our approach is the **You Only Look Once (YOLOv8)** architecture, which reformulates object detection as a single regression problem, enabling fast, end-to-end processing[cite: 497, 501, 535].

### Architecture Components:
* [cite_start]**Backbone:** **CSPDarknet53** for high-resolution feature extraction, crucial for detecting small, complex tumor structures[cite: 504].
* [cite_start]**Neck:** **PANet and Spatial Pyramid Pooling (SPP) Layers** to enhance multi-scale feature fusion and handle variations in tumor size and shape[cite: 525, 526, 527].
* [cite_start]**Head:** Responsible for predicting bounding boxes, class probabilities, and objectness scores across three different sizes to accommodate tumors of all scales[cite: 528, 529].

### Training Details:
* [cite_start]**Optimizer:** **Adam** optimizer with an initial learning rate of **0.001**[cite: 539, 540].
* [cite_start]**Loss Function:** A composite YOLO loss function combining classification, bounding box regression (confinement), and confidence (objectness) losses[cite: 541, 542].
* [cite_start]**Training:** Trained for 50 epochs with a batch size of 16[cite: 543, 544].

---

## 🛠️ Tools & Technologies

| Category | Tool / Library | Purpose |
| :--- | :--- | :--- |
| **Programming Language** | Python | [cite_start]Primary language for model development [cite: 452] |
| **Deep Learning Frameworks** | TensorFlow, Keras | [cite_start]Used for model development, training, and evaluation [cite: 453] |
| **Core Architecture** | YOLOv8 | [cite_start]The main deep learning model for object detection [cite: 463] |
| **Image Processing** | OpenCV, PIL | [cite_start]Handling image processing tasks like resizing and augmentation [cite: 454, 455] |
| **Development Environment** | Google Colab | [cite_start]Used for code execution and model training [cite: 456] |

---

## 📊 Dataset

* [cite_start]**Source:** Custom-made dataset augmented from MRI scans of Wilms tumor patients and other pediatric tumors[cite: 446].
* [cite_start]**Original Data:** Axial c portal venous phase MRI scans for 20 patients, totaling 60 images, sourced from **Radiopedia.org**[cite: 447, 448].
* [cite_start]**Augmented Data:** Expanded to nearly **1000 images** using augmentation techniques[cite: 448, 547].
* [cite_start]**Split:** The dataset was partitioned into an **8:2 ratio** for training and validation/testing[cite: 450].

---

## 🤝 Next Steps & Future Work

To further enhance the model's precision and utility, future work will focus on:

* [cite_start]**Dataset Refinement:** Organizing and better processing the dataset to yield even higher precision[cite: 563].
* [cite_start]**Feature Enrichment:** Including additional parameters and leveraging tools like **Pyradiomics** to extract more intricate features from the MRI images, potentially improving diagnostic metrics[cite: 564].
* **Wider Generalization:** Testing the model on more diverse, independent clinical datasets to further assess its real-world reliability.

---

## ✍️ Authors

This project was developed by the following researchers:

* **Venkatesh Kavididevi**
* **Necha Akhila Sri Kornepati**
* **Deepak Reddy Chelladi**
* **Shoaib Ali MD**

[cite_start]*(All authors are from the Department of Information Technology, Vardhaman College of Engineering, Hyderabad, India.)* [cite: 329, 330, 334, 336, 357, 358]
