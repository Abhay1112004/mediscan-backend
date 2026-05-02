import gdown
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import joblib

app = Flask(__name__)
CORS(app)

# ----------------------------
# Railway Safe Port (IMPORTANT)
# ----------------------------
PORT = int(os.environ.get("PORT", 5000))

# ----------------------------
# Download Function
# ----------------------------
def download_model(url, output):
    if not os.path.exists(output):
        print(f"Downloading {output}...")
        gdown.download(url, output, quiet=False)

# ----------------------------
# Download Models
# ----------------------------
download_model(
    "https://drive.google.com/uc?id=15XrXkDh7_-Y7-n7J2RKnGE6ICX0R7EgP",
    "brain_tumor_model.h5"
)

download_model(
    "https://drive.google.com/uc?id=1eiyGGubzq37REFPfxY9kyV7F7oWCPown",
    "eye_model.h5"
)

download_model(
    "https://drive.google.com/uc?id=1srNBdciReNkB38nbsFuN88a_oQkUnYKw",
    "heart_model.pkl"
)

download_model(
    "https://drive.google.com/uc?id=16KG--pPRYhivseaNhG6RGnepek9-gdxb",
    "heart_scaler.pkl"
)

# ----------------------------
# Load Models
# ----------------------------
print("Loading models...")

brain_model = tf.keras.models.load_model("brain_tumor_model.h5")
eye_model = tf.keras.models.load_model("eye_model.h5")
heart_model = joblib.load("heart_model.pkl")
heart_scaler = joblib.load("heart_scaler.pkl")

print("Models loaded successfully")

# ----------------------------
# Medical Image Validation
# ----------------------------
def is_valid_medical_image(img, scan_type):
    img_array = np.array(img)

    if scan_type == "brain":
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        color_difference = np.mean(np.abs(r - g)) + np.mean(np.abs(g - b))
        if color_difference > 25:
            return False

    elif scan_type == "eye":
        red_mean = np.mean(img_array[:, :, 0])
        green_mean = np.mean(img_array[:, :, 1])
        blue_mean = np.mean(img_array[:, :, 2])
        if red_mean < green_mean or red_mean < blue_mean:
            return False

    return True

# ----------------------------
# Classes
# ----------------------------
brain_classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

eye_classes = [
    "Mild",
    "Moderate",
    "No Diabetic Retinopathy",
    "Proliferate DR",
    "Severe"
]

# ----------------------------
# Brain Prediction
# ----------------------------
@app.route("/predict_brain", methods=["POST"])
def predict_brain():
    if "file" not in request.files:
        return jsonify({"warning": "No image uploaded"}), 400

    file = request.files["file"]

    try:
        img = Image.open(file).convert("RGB").resize((224, 224))
    except:
        return jsonify({"warning": "Invalid image"}), 400

    if not is_valid_medical_image(img, "brain"):
        return jsonify({"warning": "Invalid MRI image"}), 400

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = brain_model.predict(img_array)
    class_index = np.argmax(pred)
    confidence = float(np.max(pred)) * 100

    return jsonify({
        "prediction": brain_classes[class_index],
        "confidence": round(confidence, 2)
    })

# ----------------------------
# Eye Prediction
# ----------------------------
@app.route("/predict_eye", methods=["POST"])
def predict_eye():
    if "file" not in request.files:
        return jsonify({"warning": "No image uploaded"}), 400

    file = request.files["file"]

    try:
        img = Image.open(file).convert("RGB").resize((224, 224))
    except:
        return jsonify({"warning": "Invalid image"}), 400

    if not is_valid_medical_image(img, "eye"):
        return jsonify({"warning": "Invalid retinal image"}), 400

    img_array = np.array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    pred = eye_model.predict(img_array)
    class_index = np.argmax(pred)
    confidence = float(np.max(pred)) * 100

    return jsonify({
        "prediction": eye_classes[class_index],
        "confidence": round(confidence, 2)
    })

# ----------------------------
# Heart Prediction
# ----------------------------
@app.route("/predict_heart", methods=["POST"])
def predict_heart():
    data = request.json

    values = [
        data["age"], data["sex"], data["cp"], data["trestbps"],
        data["chol"], data["fbs"], data["restecg"], data["thalach"],
        data["exang"], data["oldpeak"], data["slope"], data["ca"], data["thal"]
    ]

    values = np.array(values).reshape(1, -1)
    values = heart_scaler.transform(values)

    pred = heart_model.predict(values)[0]
    prob = heart_model.predict_proba(values)[0][1] * 100

    result = "Heart Disease Detected" if pred == 1 else "No Heart Disease"

    return jsonify({
        "prediction": result,
        "confidence": round(prob, 2)
    })

# ----------------------------
# Run (RAILWAY FIXED)
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)