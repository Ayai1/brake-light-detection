from ultralytics import YOLO
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim

if __name__ == '__main__':
    # Load the pre-trained model from your best checkpoint
    model = YOLO("C:/YOLO_v11/runs/detect/train24/weights/best.pt")

    # Define optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)  
    # Fine-tune the model
    results = model.train(
        epochs=10,
        optimizer='AdamW',
        lr0=0.001, 
        batch=16, 
        imgsz=640,  
        augment=False, 
        workers=8,  
    )

    # Step scheduler after each epoch
    for _ in range(10):
        scheduler.step()
