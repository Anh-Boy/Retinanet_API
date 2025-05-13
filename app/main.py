from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse 
from fastapi.middleware.cors import CORSMiddleware 
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import tensorflow as tf
import numpy as np
import cv2
import os
import uvicorn 
from io import BytesIO
from PIL import Image
import base64
app = FastAPI()

# Cho phép CORS nếu bạn chạy frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- RetinaNet ---
class RetinaNetDetector:
    def __init__(self, model_path, score_thresh=0.5):
        self.model = models.load_model(model_path, backbone_name='resnet50')
        self.score_thresh = score_thresh

    def detect(self, image_array):
        image = image_array.copy()
        image_pre = preprocess_image(image.copy())
        image_pre, scale = resize_image(image_pre)
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image_pre, axis=0))
        boxes /= scale

        detections = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < self.score_thresh:
                continue
            b = box.astype(int)
            x1, y1, x2, y2 = b
            crop = image[y1:y2, x1:x2]
            detections.append({
                "crop": crop,
                "box": b,
                "label_id": int(label),
                "score": float(score)
            })
        return detections

# --- CNN ---
class CNNClassifier:
    def __init__(self, model_path, input_size=(64, 64)):
        self.model = tf.keras.models.load_model(model_path)
        self.input_size = input_size
        self.labels = [
        "Speed limit 20km/h", "Speed limit 30km/h", "Speed limit 50km/h",
        "Speed limit 60km/h", "Speed limit 70km/h", "Speed limit 80km/h",
        "End of 80km/h speed limit", "Speed limit 100km/h", "Speed limit 120km/h",
        "No overtaking", "No overtaking for trucks", "Intersection with non-priority road",
        "Start of priority road", "Yield", "Stop", "No entry", "No trucks allowed",
        "No entry (wrong way)", "General danger", "Dangerous curve left",
        "Dangerous curve right", "Double curve", "Bumpy road",
        "Slippery road", "Road narrows on the right", "Roadwork",
        "Traffic signals", "Pedestrian crossing", "Children crossing",
        "Bicycle crossing", "Slippery road (again)", "Wild animals crossing",
        "End of all speed limits", "Turn right", "Turn left", "Go straight",
        "Go straight or turn right", "Go straight or turn left",
        "Keep right", "Keep left", "Roundabout", "End of no overtaking",
        "End of no overtaking for trucks"
        ]


    def predict(self, crop):
        try:
            img = cv2.resize(crop, self.input_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            preds = self.model.predict(img)
            idx = np.argmax(preds)
            return self.labels[idx], float(preds[0][idx])
        except:
            return "Lỗi phân loại", 0.0

# --- Load models ---
retina_model_path = 'infer_model/resnet50_csv_36.h5'
cnn_model_path = 'saved_model_cnn'  # đường dẫn saved_model
retina = RetinaNetDetector(retina_model_path)
cnn = CNNClassifier(cnn_model_path)
# Thêm vào đầu file nếu chưa có
used_positions = []

def get_non_overlapping_y(y_start, height, padding=5):
    """Tìm vị trí y không bị trùng với caption đã vẽ"""
    y = y_start
    while any(abs(y - used_y) < (height + padding) for used_y in used_positions):
        y -= (height + padding)
        if y < 0:
            break
    used_positions.append(y)
    return y

def draw_caption_with_bg(image, box, caption, font_scale=0.5, thickness=1):
    b = box.astype(int)
    x1, y1 = b[0], b[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(caption, font, font_scale, thickness)
    text_w, text_h = text_size

    # Điều chỉnh y không bị đè
    y1_text = get_non_overlapping_y(y1, text_h + 4)

    # Vẽ nền đen
    cv2.rectangle(image, (x1, y1_text - text_h - 4), (x1 + text_w, y1_text), (0, 0, 0), -1)

    # Vẽ caption trắng
    cv2.putText(image, caption, (x1, y1_text - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

# --- API route ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    detections = retina.detect(image)
    results = []

    used_positions.clear()  # Reset trước mỗi ảnh

    for d in detections:
        label, conf = cnn.predict(d['crop'])
        x1, y1, x2, y2 = d['box']
        draw_box(image, d['box'], color=label_color(0))
        draw_caption_with_bg(image, d['box'], f"{label} ({conf:.2f})")
        results.append({
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "label": label,
            "confidence": round(conf, 2)
        })


    # Chuyển ảnh thành base64 để trả về
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse(content={
        "message": "Success",
        "results": results,
        "image_base64": image_base64
    })

# --- Run server ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
