# Deepfake Detection System

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
        └── real/

---

## Model

The project uses **MobileNetV2** with transfer learning.

### Input

Images are resized to:

224 × 224 × 3

### Output

The model performs binary classification:

- Fake
- Real

The trained model is stored at:

model/best_deepfake_model.keras

The model contains approximately 2.26 million parameters.

---

## Training

The model was trained using:

- MobileNetV2
- Transfer learning
- Binary classification
- Dropout regularization
- 8,000 training images
- 1,000 validation images
- 10 training epochs

The training code is available at:

model/train_model.py

---

## Model Performance

The trained model was evaluated on the test dataset containing 1,000 images.

| Metric | Score |
|---|---:|
| Accuracy | 68.50% |
| Precision | 79.37% |
| Recall | 50.00% |
| F1-Score | 61.35% |

### Confusion Matrix

```text
[[435  65]
 [250 250]]

```

### Classification Report

```text
              precision    recall  f1-score   support

fake              0.64      0.87      0.73       500
real              0.79      0.50      0.61       500

accuracy                              0.69      1000
macro avg          0.71      0.69      0.67      1000
weighted avg       0.71      0.69      0.67      1000
```

---

## Web Application

The Flask web application allows users to upload an image and receive a deepfake detection result.

### Application Workflow

```text
User
  |
  v
Upload Image
  |
  v
Flask Web Application
  |
  v
Image Preprocessing
  |
  v
MobileNetV2 Model
  |
  v
Prediction
  |
  +----> REAL
  |
  +----> FAKE
  |
  v
Confidence Score
```

The application allows users to:

1. Upload an image.
2. Process the image.
3. Run the trained deep learning model.
4. Display REAL or FAKE prediction.
5. Display the confidence percentage.

---

## Project Structure

```text
DEEPFAKE_DETECTION_SYSTEM/
│
├── model/
│   ├── best_deepfake_model.keras
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
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/23am1a3195-del/Deepfake_Detection_System.git
```

### 2. Open the project directory

```bash
cd Deepfake_Detection_System
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment on Windows

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Run:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Upload an image to receive the prediction.

---

## Testing the Model

The testing code is available at:

```text
testing/test_model.py
```

Run:

```bash
python testing/test_model.py
```

The testing script evaluates:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

---

## Preprocessing

The preprocessing code is available in:

```text
preprocessing/
```

Files:

```text
data_loader.py
inspect_dataset.py
```

`data_loader.py` loads the training, validation, and testing datasets.

`inspect_dataset.py` can be used to inspect the dataset structure and image information.

---

## Limitations

The current model achieved an accuracy of **68.50%** on the test dataset.

Therefore, the system should not be considered a perfect deepfake detector.

Possible limitations include:

- Incorrect predictions on unseen image types
- Sensitivity to image quality
- Dataset bias
- Limited training data diversity
- Difficulty detecting advanced manipulation techniques
- The current system focuses on image-based detection rather than full video analysis

---

## Future Improvements

Possible future improvements include:

- Increasing the size and diversity of the dataset
- Improving model accuracy
- Fine-tuning MobileNetV2
- Comparing multiple CNN architectures
- Adding video deepfake detection
- Adding frame-by-frame video analysis
- Adding webcam-based detection
- Improving the user interface
- Deploying the application to a cloud platform
- Adding explainable AI techniques such as Grad-CAM

---

## Disclaimer

This project is intended for **educational and research purposes**.

Deepfake detection models can produce incorrect predictions. The results should not be treated as definitive evidence without additional verification.

---

## Author

**23AM1A3195**

**Deepfake Detection System**

A deep learning project for image-based deepfake detection using MobileNetV2 and Flask.

