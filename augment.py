import cv2
import os
import random
import numpy as np
import albumentations as A
from glob import glob

# Paths
input_images_dir = "datasets/setv3p/images"  # Folder containing original images
input_labels_dir = "datasets/setv3p/labels"  # Folder containing YOLO labels
output_images_dir = "datasets/setv4p/images"
output_labels_dir = "datasets/setv4p/labels"
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Define augmentation pipeline (with bounding box support)
transform = A.Compose([
    A.MotionBlur(blur_limit=(5, 20), p=0.5),  # Simulates **fast-moving ball**
    A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3),  # Simulates **camera distortions**
    A.Perspective(scale=(0.05, 0.15), p=0.4),  # Simulates **angle changes**
    A.CLAHE(clip_limit=2, tile_grid_size=(8,8), p=0.3),  # **Boosts contrast in low light**
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Adjusts **brightness**
    A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),  # Adds **rotation jitter**
    A.HorizontalFlip(p=0.5),  # **Flips images horizontally**
    A.RandomScale(scale_limit=0.1, p=0.3),  # Randomly scales the image
    A.ISONoise(p=0.3),  # **Simulates noisy camera feeds**
    A.CoarseDropout(max_holes=2, max_height=120, max_width=120, min_holes=1, min_height=50, min_width=50, fill_value=0, p=0.5)  # Simulates **ball being blocked**
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
"""
transform = A.Compose([
    A.MotionBlur(p=0.3),  # Simulates fast movement
    A.RandomBrightnessContrast(p=0.3),  # Adjusts brightness & contrast
    A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),  # Rotates image & labels
    A.Resize(1080, 1920),  # Ensures all images are the same size
    A.RandomScale(scale_limit=0.1, p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, p=0.3),
    A.GaussianBlur(p=0.2),
    A.GridDistortion(p=0.2)
    # A.cutout(num_holes=1, max_h_size=100, max_w_size=100, fill_value=0, p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
"""

# Number of augmentations per image
num_augmentations = 8
num_occlusion_images = 2

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

        for i in range(num_augmentations):
            augmented = transform(image=image, bboxes=bboxes, category_ids=category_ids)
            aug_img = augmented["image"]
            aug_bboxes = augmented["bboxes"]

            if i < num_occlusion_images:
                for bbox in aug_bboxes:
                    x_center, y_center, width, height = bbox
                h, w, _ = aug_img.shape

                # Convert YOLO bbox format to pixel coordinates
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)

                # Generate random stripes
                num_stripes = random.randint(1, 3)  # Randomly choose number of stripes
                for _ in range(num_stripes):
                    stripe_x1 = random.randint(x1, x2 - 2)
                    stripe_x2 = random.randint(stripe_x1 + 2, x2)
                    stripe_thickness = random.randint(1, 2)

                    # Randomly choose color (black or gray)
                    color = random.randint(0, 80)

                    cv2.line(aug_img, (stripe_x1, y1), (stripe_x2, y2), (color, color, color), stripe_thickness)

            aug_img_path = os.path.join(output_images_dir, f"{img_name}_aug{i}.jpg")
            aug_label_path = os.path.join(output_labels_dir, f"{img_name}_aug{i}.txt")

            # Save augmented image
            cv2.imwrite(aug_img_path, aug_img[..., ::-1])  # Convert RGB back to BGR

            # Save augmented label
            with open(aug_label_path, "w") as f:
                for j, bbox in enumerate(aug_bboxes):
                    f.write(f"{category_ids[j]} " + " ".join(map(str, bbox)) + "\n")

print("✅ Image & Label Augmentation Complete!")

