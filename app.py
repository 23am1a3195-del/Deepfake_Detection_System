from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

MODEL_PATH = "model/best_deepfake_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

IMAGE_SIZE = (224, 224)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files.get("file")

    if file is None or file.filename == "":
        return render_template(
            "index.html",
            prediction="No image selected",
            confidence=0
        )

    image = Image.open(file).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    probability = float(
        model.predict(image_array, verbose=0)[0][0]
    )

    if probability >= 0.5:
        prediction = "REAL"
        confidence = probability * 100
    else:
        prediction = "FAKE"
        confidence = (1 - probability) * 100

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(confidence, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)