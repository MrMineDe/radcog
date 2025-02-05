import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO('radcog-0.1.4.pt')  # Replace 'best.pt' with your model file

# Open a video file or capture device (0 for webcam)
video_path = 'input/vid_out.webm'  # Replace with your video file or use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process the video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # Run YOLO inference on the frame
    results = model.predict(source=frame, conf=0.25, device=0, verbose=False)  # Adjust confidence as needed

    # Visualize detections on the frame
    annotated_frame = results[0].plot()  # Annotate the frame with detections

    # Display the frame
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

