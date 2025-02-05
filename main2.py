import time
import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO('models/radcog-0.2.1.pt')  # Replace 'best.pt' with your model file

# Open a video file or capture device (0 for webcam)
video_path = 'input/input4--2.mp4'  # Replace with your video file or use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the fps of the video
fps = cap.get(cv2.CAP_PROP_FPS) / 4
frame_delay = int(1000 / fps)

# Process the video frame by frame
while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # Run YOLO inference on the frame
    results = model.predict(source=frame, conf=0.05, device=0, verbose=False)  # Adjust confidence as needed

    # Visualize detections on the frame
    annotated_frame = results[0].plot()  # Annotate the frame with detections

    # Display the frame
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    # Maintain the correct frame rate
    elapsed_time = (time.time() - start_time) * 1000
    wait_time = max(1, frame_delay - int(elapsed_time))

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
