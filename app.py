import os, uuid
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------
# Config
# -------------------
ALLOWED_EXT = {"jpg", "jpeg", "png"}
UPLOAD_DIR = os.path.join("static", "uploads")
IMG_SIZE = (256, 256)

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "super-secret-key"  # change in prod

# -------------------
# Load model (lazy + safe)
# -------------------
MODEL_PATH = "brain_cancer_model.h5"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}")

model = None

def get_model():
    global model
    if model is None:
        model = load_model(MODEL_PATH, compile=False)
        dummy = tf.zeros((1, 256, 256, 3))
        model(dummy, training=False)  # build graph
    return model

# Match training order
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# -------------------
# Helpers
# -------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def read_rgb(path: str):
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError("Could not read uploaded image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def run_inference(img_path: str):
    rgb = read_rgb(img_path)

    # Preprocess
    in_img = cv2.resize(rgb, IMG_SIZE)
    in_arr = img_to_array(in_img)
    in_arr = preprocess_input(in_arr)
    in_arr = np.expand_dims(in_arr, axis=0)

    preds = get_model().predict(in_arr, verbose=0)[0]
    class_id = int(np.argmax(preds))

    return CLASS_NAMES[class_id], float(preds[class_id] * 100.0)

# -------------------
# Routes
# -------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            flash("Please choose an image.")
            return redirect(url_for("index"))

        f = request.files["image"]
        if f.filename == "":
            flash("No file selected.")
            return redirect(url_for("index"))

        if not allowed_file(f.filename):
            flash("Allowed types: jpg, jpeg, png")
            return redirect(url_for("index"))

        filename = secure_filename(f.filename)
        save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
        f.save(save_path)

        try:
            label, conf = run_inference(save_path)
            return render_template(
                "index.html",
                pred_label=label,
                pred_conf=f"{conf:.2f}",
                uploaded_url=url_for(
                    "static",
                    filename=f"uploads/{os.path.basename(save_path)}"
                ),
            )
        except Exception as e:
            print("Inference error:", e)
            flash(f"Inference error: {e}")
            return redirect(url_for("index"))

    return render_template("index.html", pred_label=None)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
