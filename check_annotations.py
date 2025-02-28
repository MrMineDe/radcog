import cv2
import os

# Paths to your images and labels
IMAGE_DIR = "datasets/setv10p/images/"  # Change to your image folder
LABEL_DIR = "datasets/setv10p/labels/"  # Change to your YOLO labels folder
CLASS_NAMES = ["Ball"]  # Modify this if you have multiple classes

# Load all images
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))])
index = 0  # Start from the first image

def draw_yolo_bboxes(image_path, label_path):
    """ Draws bounding boxes from YOLO labels on an image """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    h, w, _ = image.shape

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split()
                if len(data) < 5:
                    continue  # Skip invalid lines

                class_id, x, y, bw, bh = map(float, data)
                x, y, bw, bh = int(x * w), int(y * h), int(bw * w), int(bh * h)

                # Convert YOLO format to (x1, y1, x2, y2)
                x1, y1 = int(x - bw / 2), int(y - bh / 2)
                x2, y2 = int(x + bw / 2), int(y + bh / 2)

                color = (0, 255, 0)  # Green box
                label = CLASS_NAMES[int(class_id)] if int(class_id) < len(CLASS_NAMES) else f"Class {class_id}"

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image

def show_image(index):
    """ Displays image with bounding boxes """
    if 0 <= index < len(image_files):
        image_path = os.path.join(IMAGE_DIR, image_files[index])
        label_path = os.path.join(LABEL_DIR, os.path.splitext(image_files[index])[0] + ".txt")
        print("image_path:" + image_path)
        print("label_path:" + label_path)

        image = draw_yolo_bboxes(image_path, label_path)
        if image is not None:
            cv2.imshow("YOLO Label Viewer", image)

# Initial display
show_image(index)

while True:
    key = cv2.waitKey(0)

    if key == 27:  # ESC to exit
        break
    elif key == 81 or key == ord('a'):  # Left Arrow or 'A' for Previous
        index = max(0, index - 1)
    elif key == 83 or key == ord('d'):  # Right Arrow or 'D' for Next
        index = min(len(image_files) - 1, index + 1)

    show_image(index)

cv2.destroyAllWindows()
