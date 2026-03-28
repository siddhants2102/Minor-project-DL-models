# 🫁 Vital Lung AI: Smart Healthcare System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55.0-red)
![Status](https://img.shields.io/badge/Status-Active_Development-brightgreen)

**Vital Lung AI** is a clinical-grade Deep Learning web application designed to assist healthcare professionals in diagnosing respiratory diseases from Chest X-Ray images. Built as a part of my **"Smart Healthcare System"** minor project work, this tool utilizes state-of-the-art Convolutional Neural Networks (CNNs) to provide rapid, multi-class visual diagnostics.

---

## 🚀 Key Features
* **Multi-Class Classification:** Accurately distinguishes between 4 categories: `Covid-19`, `Lung Opacity`, `Viral Pneumonia`, and `Normal` lungs.
* **Transfer Learning Core:** Powered by pre-trained **DenseNet121** and **EfficientNetB0** architectures for robust feature extraction.
* **Clinical-Grade Metrics:** Optimized using Area Under the Curve (`val_auc`) and precision/recall metrics to minimize life-threatening false negatives.
* **Interactive Dashboard:** A sleek, user-friendly **Streamlit** web interface featuring real-time image processing, confidence breakdown bars, and color-coded triage alerts.
* **Imbalance Handling:** Utilizes mathematical class weighting to ensure rare diseases are properly identified without algorithmic bias.

---

## 📊 Dataset & Preprocessing
The model was trained on the **COVID-19 Radiography Database**.
* **Data Split:** Strict 80-10-10 split (Training, Validation, Testing) with fixed seeds to prevent data leakage.
* **Augmentation:** Applied dynamic `tf.keras.layers` augmentation (random flips, rotations, and zooming) to artificially expand the dataset and prevent memorization.
* **Class Weights:** Implemented custom penalty multipliers during training to address the severe imbalance between Normal X-rays and Viral Pneumonia X-rays.

The training pipeline expects a local dataset folder structured as:

```text
COVID-19_Radiography_Dataset/
├── covid-19/
├── lung_opacity/
├── normal/
└── pneumonia/
```

Each class folder should contain X-ray images (`.png`, `.jpg`, etc.).

---

## 🧠 Model Architecture
Three distinct models were engineered and evaluated:
1. **Custom Baseline CNN:** A lightweight, from-scratch architecture used to establish a learning benchmark.
2. **DenseNet121 (Champion):** Utilized pre-trained ImageNet weights with a custom classification head (`GlobalAveragePooling` -> `Dense(128)` -> `Dropout(0.25)` -> `Softmax`).
3. **EfficientNetB0:** Leveraged built-in mathematical scaling and highly efficient parameter distribution.

**Training Safeguard:**
* `EarlyStopping`: Monitored validation AUC to halt training exactly when the model plateaued, restoring peak weights.

---

## 💻 Installation & Local Setup

To run this project locally on your machine, follow these steps:

**1. Clone the repository**
```bash
git clone https://github.com/siddhants2102/Minor-project-DL-models.git
cd smart-healthcare-system
```

**2. Create a virtual environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Launch the Web Application**
```bash
streamlit run app.py
```
The application will open in your default web browser at http://localhost:8501.

**⚠️Disclaimer:** This project is for educational and research purposes only. It is not an FDA-approved medical device and should not substitute professional medical advice.