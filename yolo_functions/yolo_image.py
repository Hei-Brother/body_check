from ultralytics import YOLO
from pathlib import Path
# 加载模型
model = YOLO("../yolo_models/yolov8n.pt") # 可选模型：yolov8s.pt, yolov8m.pt（更大更准）
# results[0].show()
import os
input_dir = Path("../input_files/images/")
output_dir = Path("../output_files/images")

def process_image(img_dir):
    results = model.predict(
        source=img_dir,
        save=True,
        project=str(output_dir),
        exist_ok=True,
        name=output_dir,
        classes=[0]
    )
    # 提取检测信息
    detections = [
        {
            "class": model.names[int(box.cls)],
            "confidence": float(box.conf),
            "bbox": box.xyxy[0].tolist()
        }
        for result in results
        for box in result.boxes
    ]
    return {
        "image": img_dir,
        "detections": detections,
        "saved_to": str(output_dir)  # 输出文件的完整路径
    }
results = process_image("../input_files/images/bus.jpg")
results[0].show()