import cv2
import os
import random
import numpy as np
import albumentations as A
from glob import glob

# Paths
input_images_dir = "datasets/setv5p/images"  # Folder containing original images
input_labels_dir = "datasets/setv5p/labels"  # Folder containing YOLO labels
output_images_dir = "datasets/setv7p/images"
output_labels_dir = "datasets/setv7p/labels"
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Define augmentation pipeline (with bounding box support)
transform = A.Compose([
    A.MotionBlur(blur_limit=(5, 21), p=0.5),  # Simulates **fast-moving ball**
    A.OpticalDistortion(distort_limit=0.1, p=0.3),  # Simulates **camera distortions**
    A.Perspective(scale=(0.05, 0.15), p=0.4),  # Simulates **angle changes**
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Adjusts **brightness**
    A.Rotate(limit=200, p=0.5, border_mode=cv2.BORDER_CONSTANT),  # Adds **rotation jitter**
    A.HorizontalFlip(p=0.5),  # **Flips images horizontally**
    #A.RandomScale(scale_limit=1, p=0.3),  # Randomly scales the image
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

# Number of augmentations per image
num_augmentations = 8

# Process images & labels
image_files = glob(os.path.join(input_images_dir, "*.jpg"))  # Change extension if needed

for img_path in image_files:
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_name = os.path.basename(img_path).split(".")[0]

    label_path = os.path.join(input_labels_dir, f"{img_name}.txt")

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            bboxes = []
            category_ids = []
            for line in f.readlines():
                values = line.strip().split()
                class_id = int(values[0])
                bbox = list(map(float, values[1:]))  # YOLO format: x_center, y_center, width, height
                bboxes.append(bbox)
                category_ids.append(class_id)

        # Apply augmentations multiple times
        for i in range(num_augmentations):
            # Apply augmentation
            transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
            aug_img = transformed["image"]
            aug_bboxes = transformed["bboxes"]
            aug_category_ids = transformed["category_ids"]

            # Save augmented image
            aug_img_path = os.path.join(output_images_dir, f"{img_name}_aug{i}.jpg")
            cv2.imwrite(aug_img_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))  # Convert RGB back to BGR

            # Save augmented label
            aug_label_path = os.path.join(output_labels_dir, f"{img_name}_aug{i}.txt")
            with open(aug_label_path, "w") as f:
                for j, bbox in enumerate(aug_bboxes):
                    f.write(f"{aug_category_ids[j]} " + " ".join(map(str, bbox)) + "\n")

print("✅ Image & Label Augmentation Complete!")
