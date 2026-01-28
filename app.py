import numpy as np
import cv2
import tensorflow as tf
import base64
from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------
# CONFIG
# -------------------
ALLOWED_EXT = {"jpg", "jpeg", "png","webp"}
IMG_SIZE = (256, 256)

app = Flask(__name__)
app.secret_key = "super-secret-key"

# -------------------
# LOAD MODEL
# -------------------
MODEL_PATH = "brain_cancer_model_efficent.h5"
model = load_model(MODEL_PATH)

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ------------------- GRADCAM -------------------

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv", pred_index=None):
    base_model = model.get_layer("efficientnetb0")
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    conv_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    classifier_input = tf.keras.Input(shape=conv_model.output.shape[1:])
    x = classifier_input
    for layer in model.layers[1:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.matmul(conv_outputs, pooled_grads[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)


# ------------------- HELPERS -------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def preprocess_roi(roi):
    roi = cv2.resize(roi, IMG_SIZE)
    roi = roi.astype("float32")
    roi = preprocess_input(roi)
    roi = np.expand_dims(roi, axis=0)
    return roi


def to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


def run_inference_with_roi(frame):
    frame = cv2.resize(frame, (480, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_label = None
    detected_conf = 0
    overlayed_roi_img = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 500:
            continue

        roi = frame[y:y+h, x:x+w]
        roi_array = preprocess_roi(roi)

        preds = model.predict(roi_array)[0]
        preds = tf.nn.softmax(preds).numpy()

        class_id = np.argmax(preds)
        detected_label = CLASS_NAMES[class_id]
        detected_conf = preds[class_id] * 100

        heatmap = make_gradcam_heatmap(roi_array, model, pred_index=class_id)
        overlayed_roi_img = overlay_heatmap(cv2.resize(roi, IMG_SIZE), heatmap)

        break

    if detected_label is None:
        img = cv2.resize(frame, IMG_SIZE)
        img = preprocess_input(img.astype("float32"))
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)[0]
        preds = tf.nn.softmax(preds).numpy()

        class_id = np.argmax(preds)
        detected_label = CLASS_NAMES[class_id]
        detected_conf = preds[class_id] * 100

    return detected_label, detected_conf, overlayed_roi_img, frame


# ------------------- ROUTES -------------------

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

        try:
            file_bytes = np.frombuffer(f.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            label, conf, overlayed_roi, original_img = run_inference_with_roi(frame)

            original_b64 = to_base64(original_img)
            gradcam_b64 = to_base64(overlayed_roi) if overlayed_roi is not None else None

            return render_template(
                "index.html",
                pred_label=label,
                pred_conf=f"{conf:.2f}",
                uploaded_b64=original_b64,
                gradcam_b64=gradcam_b64
            )

        except Exception as e:
            print("Error:", e)
            flash(str(e))
            return redirect(url_for("index"))

    return render_template("index.html", pred_label=None)


if __name__ == "__main__":
    app.run(debug=True)
