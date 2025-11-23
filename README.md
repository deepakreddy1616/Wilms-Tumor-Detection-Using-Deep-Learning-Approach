# 🌟 Wilms Tumor Detection using Deep Learning (YOLOv8) 🌟

## 🔬 Project Overview

[cite_start]Wilms tumor (Nephroblastoma), often called Nephroblastoma, is a cancer seen mostly in children[cite: 22]. [cite_start]It is the most common and solid malignant neoplasm in children[cite: 23]. [cite_start]Quick and exact diagnosis of Wilms tumor is crucial for effective treatment, optimal therapeutic interventions, and improved patient outcomes[cite: 11, 24].

[cite_start]This project proposes an innovative system for Wilms tumor identification, utilizing advanced deep learning procedures[cite: 12]. [cite_start]We leverage the power of the **You Only Look Once (YOLOv8)** architecture to deliver a fast, accurate, and automated diagnostic solution for clinical applications in pediatric oncology[cite: 170, 239].

---

## ✨ Key Features

* [cite_start]**State-of-the-Art Architecture:** Utilizes the high-efficiency and accuracy of the **YOLOv8** object detection model[cite: 136, 173]. [cite_start]YOLO reformulates object detection as a single regression issue, significantly reducing calculation time and making it highly efficient[cite: 174, 175].
* [cite_start]**High Diagnostic Accuracy:** Achieved an outstanding **97.40% accuracy** on the evaluation dataset[cite: 223, 226].
* [cite_start]**Robust Performance:** The F1 score and Dice coefficient both remained at **0.97**[cite: 225, 226].
* [cite_start]**Speed and Efficiency:** YOLO's unified architecture simplifies the detection process and is especially quick, making it suitable for real-time clinical applications where convenient navigation is critical[cite: 206, 208, 244].
* [cite_start]**Preprocessing for Generalizability:** Implements comprehensive preprocessing steps including **Normalization**, **Resizing (to 416x416 pixels)**, and extensive **Augmentation** (rotation, flipping, zooming) to prevent overfitting and enhance the model's robustness and ability to generalize to unseen data[cite: 131, 132, 133, 134, 242].

---

## 🚀 Performance Metrics

[cite_start]Our model was rigorously evaluated on an augmented dataset of nearly 1000 MRI images[cite: 220]. [cite_start]The results highlight the superiority of our deep learning model in terms of accuracy, sensitivity, and particularity[cite: 17, 223].

| Metric | Value |
| :--- | :--- |
| **Accuracy** | [cite_start]**97.40%** [cite: 226] |
| Recall / Sensitivity | [cite_start]98.26% [cite: 226] |
| Precision | [cite_start]96.55% [cite: 226] |
| **F1 Score** | [cite_start]**0.97** [cite: 226] |
| **Dice Coefficient** | [cite_start]**0.97** [cite: 226] |

---

## ⚙️ Methodology

[cite_start]The methodology focuses on meticulously planned deep learning techniques trained on a diverse dataset of clinical imaging scans[cite: 13].

### Model Architecture (YOLOv8)

[cite_start]The primary model utilized for Wilms tumor location is the You Only Look Once (YOLO) architecture, explicitly YOLOv8[cite: 136].

* [cite_start]**Backbone: CSPDarknet53** : Serves as the feature extractor, providing high-resolution feature maps necessary for identifying small and complex structures like tumors[cite: 177]. [cite_start]This network is planned with Cross-Stage Partial (CSP) associations, enhancing gradient flow[cite: 178].
* [cite_start]**Neck: PANet and SPP Layers** : Integrates **PANet** (Path Aggregation Network) to improve data flow for better component combination, and **Spatial Pyramid Pooling (SPP)** layers to help maintain spatial information and handle varieties in tumor size and shape[cite: 198, 199, 200].
* **Head:** The location head is liable for anticipating bouncing boxes, class probabilities, and objectness scores. [cite_start]It yields three sizes of forecasts, accommodating tumors of different sizes and guaranteeing exact limitation and grouping[cite: 201, 202].

### Training Process

* [cite_start]**Optimizer:** The **Adam** streamlining agent is utilized with an underlying learning rate of **0.001**[cite: 212]. [cite_start]Adam's versatile learning rate capacities make it suitable for training deep neural networks[cite: 213].
* [cite_start]**Loss Function:** The YOLO loss function is a composite of classification loss, confinement loss (bounding box relapse), and confidence loss (objectness score), ensuring balanced enhancement[cite: 214, 215].
* [cite_start]**Epochs and Batch Size:** The model is trained for **50 epochs** [cite: 216][cite_start], with a group size of **16**[cite: 217]. [cite_start]Early halting is used to forestall overfitting[cite: 216].

### Post-Processing
* [cite_start]**Non-Greatest Concealment (NMS):** Applied to the anticipated bounding boxes to eliminate duplicate detection and hold the most certain forecasts[cite: 167].
* [cite_start]**Morphological Operations:** Utilized to refine the edges of the bounding boxes, working on the precision of tumor confinement[cite: 168].

---

## 📊 Dataset

* [cite_start]**Source:** A custom-made dataset was used, consisting of MRI scans of Wilms tumor patients and patients suffering from other pediatric tumors[cite: 119].
* [cite_start]**Original Data Collection:** The images of the MRI scans were taken from **Radiopedia.org**, a non-profit organization[cite: 120, 171].
* [cite_start]**Data Size:** Considered a total of **20 patients' data**, consisting of **60 Axial c portal venous phase MRI scans**[cite: 121, 172].
* [cite_start]**Augmentation:** The images were later augmented to expand the dataset to near **1000 images**[cite: 121, 172].
* [cite_start]**Split:** The dataset is partitioned into an **8:2 ratio** for testing to validation[cite: 123].

---

## 🛠️ Tools & Technologies

| Category | Tool / Library | Purpose |
| :--- | :--- | :--- |
| **Programming Language** | Python | [cite_start]Used for creating and executing the profound learning models[cite: 125]. |
| **Deep Learning Frameworks** | TensorFlow and Keras | [cite_start]Utilized for model turn of events, preparing, and assessment[cite: 126]. |
| **Image Processing** | OpenCV | [cite_start]A library utilized for various image handling tasks, such as resizing, augmentation, and post-processing[cite: 127]. |
| **Development Environment** | Google Colab | [cite_start]A code execution notebook utilized to run and execute the created code with the model and dataset[cite: 129]. |
| **Utility** | PIL | [cite_start]Used to convert image extensions to jpeg[cite: 128]. |

---

## 🤝 Next Steps & Future Work

[cite_start]Future endeavors will focus on biomarkers and further improving outcomes for patients[cite: 86].

* [cite_start]**Data Organization:** The dataset can be more organized and well-processed to yield better results with this model[cite: 236].
* [cite_start]**Feature Extraction:** Many other parameters can be included, and usage of **pyradiomics** will also help to extract many features out of the MRI images, this helps in increasing the metrics of the developed model[cite: 237].
* [cite_start]**Generalizability:** The model's generalizability is assessed using an independent test set, providing a fair assessment of execution[cite: 165].

---

## ✍️ Authors

This project was developed by the following researchers:

* [cite_start]**Venkatesh Kavididevi** [cite: 2]
* [cite_start]**Necha Akhila Sri Kornepati** [cite: 7]
* [cite_start]**Deepak Reddy Chelladi** [cite: 9]
* [cite_start]**Shoaib Ali MD** [cite: 30]

[cite_start]*(All authors are from the Department of Information Technology, Vardhaman College of Engineering, Hyderabad, India [cite: 3, 31, 10, 32]).*
