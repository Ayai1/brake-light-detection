from ultralytics import YOLO
model = YOLO('C:/YOLO_v11/runs/detect/train15/weights/best.pt')
results = model.predict(source="C:/YOLO_v11/test_image.jpg.jpg", conf=0.25, save=True)
for result in results:
    result.show() 
