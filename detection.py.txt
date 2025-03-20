from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO('C:/YOLO_v11/runs/detect/train11/weights/best.pt')  # Use forward slashes or double backslashes for paths

# Run inference on an image
results = model.predict(source='C:/YOLO_v11/test_image.jpg', conf=0.2)

# Check detections
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])  # Class ID
        conf = box.conf[0]  # Confidence score
        if cls == 1:
            print(f"Brake lights ON detected! (Confidence: {conf:.2f})")
        else:
            print(f"Brake lights OFF detected. (Confidence: {conf:.2f})")
