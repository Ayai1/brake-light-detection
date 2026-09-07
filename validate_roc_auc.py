import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from ultralytics import YOLO

def main():
    # Load your trained YOLO model
    model = YOLO('C:/YOLO_v11/runs/detect/train25/weights/best.pt')

    # Paths to validation images and labels
    val_images = 'C:/YOLO_v11/val/images'  
    val_labels = 'C:/YOLO_v11/val/labels' 

    y_true = []
    y_scores = []

    # Run predictions on the validation images
    results = model.predict(source=val_images, conf=0.01)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Extract predicted class
            conf = box.conf[0].cpu().numpy()  # Extract confidence score

            # Assuming binary classification: 0 = car_brakeOFF, 1 = car_brakeON
            y_true.append(cls)
            y_scores.append(conf)

    # Convert lists to NumPy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
