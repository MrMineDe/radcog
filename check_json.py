import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

def load_annotations(json_file):
    """Load the JSON annotations from Label Studio."""
    with open(json_file, 'r') as file:
        annotations = json.load(file)
    return annotations

def visualize_annotations(video_path, annotations):
    """Visualize annotations over the video frame-by-frame."""
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened correctly
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize frame index
    frame_idx = 0

    while True:
        # Read the current frame
        ret, frame = cap.read()

        if not ret:
            break

        # Overlay the annotations (bounding boxes) on the frame
        for annotation in annotations:
            for box in annotation['box']:
                for seq in box['sequence']:
                    if seq['frame'] == frame_idx + 1:  # Check if the frame corresponds to the annotation
                        # Assuming values are normalized between 0 and 100
                        x = seq['x'] * video_width / 100  # Convert to pixels
                        y = seq['y'] * video_height / 100  # Convert to pixels
                        width = seq['width'] * video_width / 100  # Convert to pixels
                        height = seq['height'] * video_height / 100  # Convert to pixels
                        
                        if seq['enabled']:  # Only draw if the annotation is enabled
                            # Draw rectangle and add the label
                            cv2.rectangle(frame, 
                                          (int(x), int(y)), 
                                          (int(x + width), int(y + height)), 
                                          (0, 255, 0), 2)
                            label = f"Frame: {frame_idx+1}, Time: {seq['time']:.2f}s"
                            cv2.putText(frame, label, (int(x), int(y)-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame from BGR (OpenCV format) to RGB (matplotlib format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame
        plt.imshow(frame_rgb)
        plt.axis('off')
        plt.draw()

        # Wait for the user to press a key to navigate between frames
        key = plt.waitforbuttonpress(timeout=-1)  # Wait indefinitely for key press
        
        if key is False:  # Left mouse click or arrow keys can be used to go forward/backward
            # Right arrow or any key goes forward
            frame_idx = min(frame_idx + 1, total_frames - 1)
        elif key is True:  # Left mouse click or arrow keys can be used to go forward/backward
            # Left arrow goes backward
            frame_idx = max(frame_idx - 1, 0)

        # Exit if 'Q' is pressed
        if plt.waitforbuttonpress() == 'q':
            break

        # Break the loop if the frame index reaches the last frame
        if frame_idx >= total_frames - 1:
            break

    # Release the video capture object
    cap.release()
    plt.close()

if __name__ == '__main__':
    # Replace these paths with your actual file paths
    json_file = 'annotated_video.json'  # Path to your exported JSON_MIN file from Label Studio
    video_path = '/home/mrmine/.local/share/label-studio/media/upload/2/0b7e7c71-input4--1.mp4' # Path to your video file

    # Load annotations from the JSON file
    annotations = load_annotations(json_file)

    # Visualize the annotations on the video with manual frame navigation
    visualize_annotations(video_path, annotations)
