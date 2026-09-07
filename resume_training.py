from ultralytics import YOLO

if __name__ == '__main__':
    # Load the model checkpoint
    model = YOLO("C:/YOLO_v11/runs/detect/train15/weights/last.pt")

    # Continue training from this checkpoint
    model.train(
        workers=8,   
        batch=10,     
        epochs=50,    
        imgsz=640,    
    )
