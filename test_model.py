import tensorflow as tf
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# SETTINGS
# -----------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

DATASET_PATH = "dataset/8020"
MODEL_PATH = "model/best_deepfake_model.keras"

# -----------------------------
# LOAD TEST DATA
# -----------------------------
test_data = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/test",
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_data.class_names

print("Classes:", class_names)

# -----------------------------
# LOAD BEST MODEL
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

print("\nBest model loaded successfully!")

# -----------------------------
# PREDICTIONS
# -----------------------------
y_true = []
y_probability = []

for images, labels in test_data:
    predictions = model.predict(images, verbose=0)

    y_true.extend(labels.numpy().flatten())
    y_probability.extend(predictions.flatten())

y_true = np.array(y_true)
y_probability = np.array(y_probability)

# Convert probabilities to classes
y_pred = (y_probability >= 0.5).astype(int)

# -----------------------------
# METRICS
# -----------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n" + "=" * 50)
print("DEEPFAKE DETECTION TEST RESULTS")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall   : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-Score : {f1:.4f} ({f1 * 100:.2f}%)")

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)

print("=" * 50)