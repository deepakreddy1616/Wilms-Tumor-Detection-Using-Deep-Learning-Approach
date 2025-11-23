# 🌟 Wilms Tumor Detection using Deep Learning (YOLOv8) 🌟

## 🔬 Project Overview

Wilms tumor (Nephroblastoma) is a cancer seen mostly in children.  
It is the most common solid malignant neoplasm in children.  
Quick and exact diagnosis of Wilms tumor is crucial for effective treatment, optimal therapeutic interventions, and improved patient outcomes.

This project proposes an innovative system for Wilms tumor identification, utilizing advanced deep learning procedures.  
We leverage the power of the **You Only Look Once (YOLOv8)** architecture to deliver a fast, accurate, and automated diagnostic solution for clinical applications in pediatric oncology.

---

## ✨ Key Features

* **State-of-the-Art Architecture:** Utilizes the high-efficiency and accuracy of the **YOLOv8** object detection model. YOLO reformulates object detection as a single regression issue, significantly reducing calculation time and making it highly efficient.
* **High Diagnostic Accuracy:** Achieved an outstanding **97.40% accuracy** on the evaluation dataset.
* **Robust Performance:** The F1 score and Dice coefficient both remained at **0.97**.
* **Speed and Efficiency:** YOLO's unified architecture simplifies the detection process and is especially quick, making it suitable for real-time clinical applications.
* **Preprocessing for Generalizability:** Implements comprehensive preprocessing steps including **Normalization**, **Resizing (to 416x416 pixels)**, and extensive **Augmentation** (rotation, flipping, zooming) to prevent overfitting and enhance the model's robustness.

---

## 🚀 Performance Metrics

Our model was rigorously evaluated on an augmented dataset of nearly 1000 MRI images.  
The results highlight the superiority of our deep learning model in terms of accuracy, sensitivity, and particularity.

| Metric               | Value       |
|----------------------|------------|
| **Accuracy**         | **97.40%** |
| Recall / Sensitivity | 98.26%     |
| Precision            | 96.55%     |
| **F1 Score**         | **0.97**   |
| **Dice Coefficient** | **0.97**   |

---

## ⚙️ Methodology

The methodology focuses on meticulously planned deep learning techniques trained on a diverse dataset of clinical imaging scans.

### Model Architecture (YOLOv8)

The primary model utilized for Wilms tumor location is the You Only Look Once (YOLO) architecture, specifically YOLOv8.

* **Backbone: CSPDarknet53** : Serves as the feature extractor, providing high-resolution feature maps necessary for identifying small and complex structures like tumors. This network uses Cross-Stage Partial (CSP) connections, enhancing gradient flow.
* **Neck: PANet and SPP Layers** : Integrates **PANet** (Path Aggregation Network) to improve data flow for better component combination, and **Spatial Pyramid Pooling (SPP)** layers to help maintain spatial information and handle variety in tumor size and shape.
* **Head:** Responsible for outputting bounding boxes, class probabilities, and objectness scores. Provides predictions at three sizes, accommodating tumors of different sizes.

### Training Process

* **Optimizer:** Adam with initial learning rate **0.001**.
* **Loss Function:** The YOLO loss combines classification loss, localization loss (bounding box regression), and confidence (objectness score) loss.
* **Epochs and Batch Size:** Trained for **50 epochs**, with batch size **16**. Early stopping used to prevent overfitting.

### Post-Processing
* **Non-Maximum Suppression (NMS):** Applied to predicted bounding boxes to eliminate duplicates.
* **Morphological Operations:** Used to refine bounding box precision.

---

## 📊 Dataset

* **Source:** Custom MRI dataset of Wilms tumor patients and other pediatric tumors.
* **Original Data Collection:** MRI scans from Radiopedia.org.
* **Data Size:** 20 patients; 60 axial c portal venous phase MRI scans.
* **Augmentation:** Expanded original images to nearly 1000 images through rotation, flipping, and zooming.
* **Split:** 80:20 ratio for train and validation.

---

## 🛠️ Tools & Technologies

| Category                | Tool / Library      | Purpose                                    |
|-------------------------|--------------------|---------------------------------------------|
| Programming Language    | Python             | Creating and executing the deep learning models |
| Deep Learning Frameworks| TensorFlow, Keras  | Model development, training, and evaluation |
| Object Detection        | YOLOv8 (Ultralytics)| Fast real-time detection                    |
| Image Processing        | OpenCV, PIL        | Preprocessing, augmentation                |
| Development Environment | Google Colab       | Notebook-based code execution               |

---

## 🤝 Next Steps & Future Work

Future work will focus on exploring biomarkers and further improvements in outcomes.

* **Data Organization:** Improve data organization for better results.
* **Feature Extraction:** Use **pyradiomics** to extract more image features.
* **Generalizability:** Assess using an independent test set.


---

