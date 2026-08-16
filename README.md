# Deepfake Detection System

A deep learning based web application for detecting whether an image is REAL or FAKE.

## Project Overview

This project uses a deep learning model based on MobileNetV2 to classify images as real or deepfake.

The trained model is integrated with a Flask web application where users can upload an image and receive a prediction with a confidence score.

## Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Pillow
- Flask
- HTML/CSS

## Dataset

The dataset contains:

- 8,000 training images
- 1,000 validation images
- 1,000 testing images

Each dataset is divided into:

- Fake
- Real

The dataset is not included in this repository because of its large size.

## Model

The project uses MobileNetV2 with transfer learning.

Input image size:

224 × 224 pixels

The final layer performs binary classification:

- 0 → Fake
- 1 → Real

## Model Performance

Test results:

| Metric | Score |
|---|---:|
| Accuracy | 68.50% |
| Precision | 79.37% |
| Recall | 50.00% |
| F1-Score | 61.35% |

## Web Application

The Flask application allows users to:

1. Upload an image.
2. Process the image.
3. Run the trained deep learning model.
4. Display REAL or FAKE prediction.
5. Display the confidence percentage.

## Project Structure

```text
DEEPFAKE_DETECTION_SYSTEM/
│
├── model/
│   └── train_model.py
│
├── preprocessing/
│   ├── data_loader.py
│   └── inspect_dataset.py
│
├── testing/
│   └── test_model.py
│
├── templates/
│   └── index.html
│
├── static/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md