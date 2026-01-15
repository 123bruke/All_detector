# All_detector
This projects is detect objects by pretrain ML model 

This project is a **Flask-based AI web application** that allows users to upload images and automatically detect objects using **YOLOv8** and **ResNet18**.

The system can identify:
- 👤 Humans
- 🐶 Animals
- ⚠️ Dangerous objects
- 📦 General objects

No training is required. The models are **pretrained** and used only for prediction.
## feture of this projects 

- Upload image from browser
- Object detection using YOLOv8
- Image classification using ResNet18
- Bounding boxes drawn on detected objects
- User registration and login
- Image history stored in SQL database
- Clean UI (HTML, CSS, JavaScript)

-
## 🧠 AI Models Used

### 1. YOLOv8
- Purpose: Object Detection
- Dataset: COCO (pretrained)
- File: `yolov8n.pt`

### 2. ResNet18
- Purpose: Image Classification
- Dataset: ImageNet (pretrained)
