from ultralytics import YOLO
def validate_model():
    # Load the trained model
    model = YOLO(r"C:\YOLO_v11\runs\detect\train18\weights\best.pt")

    # Run validation and plot confusion matrix
    results = model.val(data="C:\YOLO_v11\dataset_custom.yaml", save_conf=True)
    results.plot()  # Save the confusion matrix

if __name__ == '__main__':
    validate_model()
