from ultralytics import YOLO
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim

# Optional: Import Albumentations if you need custom augmentations
from albumentations import Compose, RandomBrightnessContrast, GaussianBlur, MotionBlur, CLAHE, RandomCrop
from albumentations.pytorch import ToTensorV2

# 1. Define a custom augmentation pipeline (optional)
def get_augmentation_pipeline():
    return Compose([
        RandomBrightnessContrast(p=0.5),
        GaussianBlur(blur_limit=5, p=0.3),
        MotionBlur(blur_limit=5, p=0.3),
        CLAHE(clip_limit=2.0, p=0.3),
        RandomCrop(height=480, width=640, p=0.4),
        ToTensorV2(),
    ])

if __name__ == '__main__':
    # 2. Load YOLO model
    model = YOLO('yolo11m.pt')  # Load pre-trained YOLOv11 model

    # 3. Initial Training Phase
    results = model.train(
        data="C:/YOLO_v11/dataset_custom.yaml",  # Dataset config
        imgsz=640,
        batch=16,
        epochs=100,  # Train for more epochs
        workers=8,
        device=0,  # Use GPU
        augment=True,  # Built-in augmentations
        conf=0.5  # Confidence threshold
    )

    # 4. Fine-tuning Phase with Learning Rate Scheduler
    optimizer = optim.Adam(model.model.parameters(), lr=0.001)  # Custom optimizer
    scheduler = CosineAnnealingLR(optimizer, T_max=10)  # Learning rate scheduler

    for epoch in range(10):  # Fine-tune for 10 additional epochs
        model.train
