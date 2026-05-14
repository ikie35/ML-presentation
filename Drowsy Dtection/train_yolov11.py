from ultralytics import YOLO

def main():
    # Load a pretrained YOLOv11 nano model
    model = YOLO('yolo11n.pt')

    # Train the model
    print("Starting training for driver drowsiness detection...")
    results = model.train(
        data='/Users/crischanlarita/Documents/Thesis/Drowsy Dtection/Thesis v5i Yolov11/data.yaml',
        epochs=50,       # Adjust epochs as needed
        imgsz=640,       # Image size
        batch=16,        # Batch size
        name='drowsiness_model', # Project name to save weights
        device='mps'     # Standard for Mac M-series. Use 'cpu' if no Apple Silicon
    )
    
    print("Training complete! Model saved to runs/detect/drowsiness_model/")

if __name__ == '__main__':
    main()
