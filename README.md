# 手写数字识别网站

这是一个基于 Flask、PyTorch 和 MNIST 卷积神经网络模型的手写数字识别网站。项目用于识别 `0–9` 的单个手写数字，提供单张图片识别、批量图片识别和网页手写输入识别三项功能。

## 功能

### 单张图片识别

- 上传一张 PNG 格式的手写数字图片
- 使用训练好的模型进行识别
- 显示图片缩略图和预测数字
- 当文件名中包含数字标签时，显示预测是否正确

### 批量图片识别

- 一次上传多张 PNG 格式的手写数字图片
- 分别显示每张图片的预测结果
- 从文件名中读取数字作为真实标签
- 对带有标签的图片计算整体准确率

例如，文件名 `sample_7.png` 中的 `7` 会被作为真实标签。

### 网页手写输入识别

- 在网页画布中用鼠标或触摸操作书写数字
- 点击识别按钮后立即显示预测结果
- 支持一键清空画布并重新书写

## 技术栈

- Flask：网站和识别接口
- PyTorch / torchvision：模型加载和图片预处理
- Pillow：图片读取与转换
- HTML、CSS、JavaScript Canvas：网页界面和手写输入
- Gunicorn：线上部署

## 项目结构

```text
handwriting-recognition/
├── app.py                         # Flask 应用与识别逻辑
├── models/MNISTMODELV0.pth        # 训练好的识别模型
├── templates/                     # 网页模板
├── static/style.css               # 页面样式
├── requirements.txt               # Python 依赖
├── render.yaml                    # Render 部署配置
└── 手写字体识别网站功能规划.md      # 功能说明
```

## 本地运行

建议使用 Python 3.11。

```bash
pip install -r requirements.txt
python app.py
```

启动后访问：<http://127.0.0.1:5000/>

## 页面地址

- `/`：首页
- `/upload`：单张图片识别
- `/batch`：批量图片识别
- `/draw`：网页手写输入识别

## 使用限制

- 目前仅支持 PNG 图片
- 模型用于识别 `0–9` 的单个手写数字，不支持汉字、字母或多位数字
- 批量准确率依赖文件名中的数字标签
- 识别效果与笔迹大小、位置、背景和训练数据的相似程度有关

## 部署

仓库包含 Render 配置文件。连接 Render 后可使用以下命令部署：

- Build Command：`pip install -r requirements.txt`
- Start Command：`gunicorn app:app`
