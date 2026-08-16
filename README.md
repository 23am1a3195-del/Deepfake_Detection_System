**# Deepfake Detection System

A deep learning based web application for detecting whether an uploaded image is **REAL** or **FAKE**.

This project uses a **MobileNetV2-based deep learning model** with transfer learning. The trained model is integrated into a **Flask web application**, allowing users to upload an image and receive a prediction with a confidence score.

---

## Project Overview

Deepfake images are digitally manipulated images that can be difficult to distinguish from authentic images.

This project aims to detect whether an input image is real or fake using a deep learning image classification model.

The system contains:

- Dataset preprocessing
- Deep learning model training
- Model evaluation
- Flask web application
- Image upload functionality
- REAL/FAKE prediction
- Confidence score

---

## Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- Flask
- NumPy
- Pillow
- OpenCV
- Pandas
- Scikit-learn
- HTML
- CSS

---

## Dataset

The dataset contains **10,000 images** divided into training, validation, and testing sets.

### Dataset Distribution

| Dataset | Fake | Real | Total |
|---|---:|---:|---:|
| Training | 4,000 | 4,000 | 8,000 |
| Validation | 500 | 500 | 1,000 |
| Testing | 500 | 500 | 1,000 |
| **Total** | **5,000** | **5,000** | **10,000** |

The dataset is organized into:

```text
dataset/
└── 8020/
    ├── train/
    │   ├── fake/
    │   └── real/
    │
    ├── valid/
    │   ├── fake/
    │   └── real/
    │
    └── test/
        ├── fake/
        └── real/**
