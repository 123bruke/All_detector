import torch
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.eval()
labels = models.ResNet18_Weights.DEFAULT.meta["categories"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

yolo = YOLO("models/yolov8n.pt")

def predict_resnet(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = resnet(tensor)
        probs = torch.nn.functional.softmax(out, dim=1)
        idx = out.argmax(1).item()
        confidence = probs[0, idx].item()
    return {"label": labels[idx], "confidence": confidence}

def predict_yolo(image_path, display=False):
    results = yolo(image_path)[0]
    detections = []
    img = cv2.imread(image_path)
    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])
        conf = float(box[4])
        cls_idx = int(box[5])
        label = results.names[cls_idx]
        detections.append({"label": label, "confidence": conf, "box": [x1, y1, x2, y2]})
        if display:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    if display:
        cv2.imshow("YOLO Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return detections

def predict_image(image_path, display=False):
    return {
        "resnet": predict_resnet(image_path),
        "yolo": predict_yolo(image_path, display=display)
    }

def predict_webcam(display=True):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, frame)
        results = predict_image(temp_path, display=False)
        for det in results["yolo"]:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]
            conf = det["confidence"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        res_label = results["resnet"]["label"]
        res_conf = results["resnet"]["confidence"]
        cv2.putText(frame, f"ResNet: {res_label} {res_conf:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("AI Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "example.jpg"
    predict_image(image_path, display=True)
