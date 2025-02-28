import cv2
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO("models/radcog-0.5.5.pt")  

# Video source (0 = webcam, or replace with 'video.mp4')
video_source = "input/input4--2.mp4"  # Change to 0 for webcam

# Open video capture
cap = cv2.VideoCapture(video_source)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define output video writer
out = cv2.VideoWriter("output_tracked.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Run YOLO tracking
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")  

    # Draw tracked objects
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # Get bounding boxes
        ids = r.boxes.id.int().cpu().numpy() if r.boxes.id is not None else []  # Get tracking IDs

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"Radball {track_id}"  # Label with tracking ID
            color = (0, 255, 0)  # Green

            # Draw bounding box & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the frame live
    cv2.imshow("YOLO Tracking", frame)
    out.write(frame)  # Save frame to output video

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
