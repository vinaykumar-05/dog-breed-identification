import os
import json
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ----------------------------
# Flask Setup
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Load Model
# ----------------------------
model = load_model("dogbreed_model.keras")

# ----------------------------
# Load Class Names
# ----------------------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ----------------------------
# Home Route
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ----------------------------
# Prediction Route
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No File Selected"

    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # Image Preprocess
    img = image.load_img(filepath, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    pred = model.predict(img_array)

    confidence = np.max(pred)
    predicted_class = class_names[np.argmax(pred)]

    if confidence < 0.3:
        predicted_class = "Not a Dog Image"
    else:
        predicted_class = f"{predicted_class} ({confidence*100:.2f}%)"

    return render_template(
        "result.html",
        prediction=predicted_class,
        img_path=filepath
    )

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
