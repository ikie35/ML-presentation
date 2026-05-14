import cv2
from ultralytics import YOLO

def main():
    # 1. Load your newly trained model
    # Note: If your training isn't completely finished yet, this file might not exist!
    # Update the path if your training run saved the weights somewhere else.
    model_path = 'runs/detect/drowsiness_model/weights/best.pt'
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Could not load model at {model_path}. Did the training finish completely?")
        print(e)
        return

    # 2. Open the MacBook built-in camera
    # On Macs, 0 is typically the built-in FaceTime HD Camera. 
    # Try 1 or 2 if you have an external webcam plugged in.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    print("Starting webcam... Press 'q' to quit.")

    while True:
        # Read the next frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab a frame from the camera.")
            break

        # 3. Run YOLO inference on the frame
        # verbose=False prevents the terminal from getting spammed with text
        results = model(frame, verbose=False)

        # 4. Get the image with bounding boxes drawn on it
        annotated_frame = results[0].plot()

        # 5. Display the video feed on the screen
        cv2.imshow('Driver Drowsiness Detection', annotated_frame)

        # Wait for 1 millisecond; if the 'q' key is pressed, break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. Clean up resources when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
