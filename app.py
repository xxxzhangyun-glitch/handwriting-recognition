import base64
import os
import re
from io import BytesIO

import numpy as np
import onnxruntime as ort
from flask import Flask, flash, jsonify, redirect, render_template, request, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "local-development-key")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"png"}
MODEL_PATH = os.path.join(BASE_DIR, "models", "MNISTMODELV0.onnx")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        MODEL_PATH,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


model = load_model()
MODEL_INPUT_NAME = model.get_inputs()[0].name
MODEL_OUTPUT_NAME = model.get_outputs()[0].name


def preprocess_image(image, from_canvas=False):
    image = image.convert("L").resize((28, 28), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32) / 255.0

    # 上传图片通常是白底黑字，需要反色为 MNIST 的黑底白字。
    # 网页画布本身已经是黑底白字，因此无需反色。
    if not from_canvas:
        pixels = 1.0 - pixels

    return pixels[np.newaxis, np.newaxis, :, :]


def predict(image, from_canvas=False):
    input_tensor = preprocess_image(image, from_canvas=from_canvas)
    output = model.run([MODEL_OUTPUT_NAME], {MODEL_INPUT_NAME: input_tensor})[0]
    return int(np.argmax(output, axis=1)[0])


def predict_image(img_path):
    with Image.open(img_path) as image:
        return predict(image)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/draw", methods=["GET", "POST"])
def draw():
    if request.method == "POST":
        data_url = request.form.get("image", "")
        if "," not in data_url:
            return jsonify({"error": "无效的图片数据"}), 400

        try:
            encoded = data_url.split(",", 1)[1]
            image = Image.open(BytesIO(base64.b64decode(encoded)))
            prediction = predict(image, from_canvas=True)
        except Exception:
            app.logger.exception("手写图片识别失败")
            return jsonify({"error": "识别失败，请重新书写后再试"}), 400

        return jsonify({"pred": prediction})

    return render_template("draw.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            flash("未选择文件")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("未选择文件")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            prediction = predict_image(file_path)

            match = re.search(r"(\d+)", filename)
            label = int(match.group(1)) if match else None
            is_correct = (label == prediction) if label is not None else None
            accuracy = 1.0 if is_correct else 0.0 if is_correct is not None else None

            return render_template(
                "recognize.html",
                filename=filename,
                pred=prediction,
                is_correct=is_correct,
                acc=accuracy,
            )

        flash("仅支持PNG图片")
        return redirect(request.url)

    return render_template("upload.html")


@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "POST":
        files = request.files.getlist("files")
        if not files or files[0].filename == "":
            flash("未选择文件")
            return redirect(request.url)

        results = []
        correct = 0
        total = 0

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                prediction = predict_image(file_path)

                match = re.search(r"(\d+)", filename)
                label = int(match.group(1)) if match else None
                is_correct = (label == prediction) if label is not None else None

                if is_correct is not None:
                    total += 1
                    if is_correct:
                        correct += 1

                results.append(
                    {
                        "filename": filename,
                        "pred": prediction,
                        "label": label,
                        "is_correct": is_correct,
                    }
                )

        accuracy = (correct / total) if total > 0 else None
        return render_template("batch_test.html", results=results, acc=accuracy)

    return render_template("batch_test.html", results=None, acc=None)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
