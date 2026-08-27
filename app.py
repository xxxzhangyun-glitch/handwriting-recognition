import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'local-development-key')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png'}
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'MNISTMODELV0.pth')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 加载模型结构
class MNISTCNN(torch.nn.Module):
    def __init__(self, output_shape: int):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.25),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.25),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 7 * 7, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, output_shape)
        )
    def forward(self, x):
        return self.layer_stack(x)

def load_model():
    model = MNISTCNN(10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x)
])

def predict_image(img_path):
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        pred = output.argmax(dim=1).item()
    return pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/draw', methods=['GET', 'POST'])
def draw():
    if request.method == 'POST':
        # 接收base64图片
        import base64
        data_url = request.form['image']
        header, encoded = data_url.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        from io import BytesIO
        img = Image.open(BytesIO(img_bytes))
        img = img.convert('L')  # 灰度
        img = img.resize((28, 28))
        img = Image.eval(img, lambda x: 255 - x)  # 反色
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1).item()
        return jsonify({'pred': pred})
    return render_template('draw.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('未选择文件')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('未选择文件')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            pred = predict_image(file_path)
            # 新增：比对标签
            import re
            match = re.search(r'(\d+)', filename)
            label = int(match.group(1)) if match else None
            is_correct = (label == pred) if label is not None else None
            acc = 1.0 if is_correct else 0.0 if is_correct is not None else None
            return render_template('recognize.html', filename=filename, pred=pred, is_correct=is_correct, acc=acc)
        else:
            flash('仅支持PNG图片')
            return redirect(request.url)
    return render_template('upload.html')

@app.route('/batch', methods=['GET', 'POST'])
def batch():
    if request.method == 'POST':
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            flash('未选择文件')
            return redirect(request.url)
        results = []
        correct = 0
        total = 0
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                pred = predict_image(file_path)
                # 从文件名提取标签
                import re
                match = re.search(r'(\d+)', filename)
                label = int(match.group(1)) if match else None
                is_correct = (label == pred) if label is not None else None
                if is_correct is not None:
                    total += 1
                    if is_correct:
                        correct += 1
                results.append({'filename': filename, 'pred': pred, 'label': label, 'is_correct': is_correct})
        acc = (correct / total) if total > 0 else None
        return render_template('batch_test.html', results=results, acc=acc)
    return render_template('batch_test.html', results=None, acc=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG') == '1')
