import os
import shutil
import random

# Paths
images_dir = '/home/mrmine/prg/radcog/datasets/setv11p/images/'  # Directory containing all images
labels_dir = '/home/mrmine/prg/radcog/datasets/setv11p/labels/'  # Directory containing all labels
output_dir = '/home/mrmine/prg/radcog/datasets/setv11'  # Directory where train/val subdirectories will be created
train_split = 0.8  # 80% for training

# Ensure output subdirectories exist
train_images_dir = os.path.join(output_dir, 'train/images/')
val_images_dir = os.path.join(output_dir, 'val/images/')
train_labels_dir = os.path.join(output_dir, 'train/labels/')
val_labels_dir = os.path.join(output_dir, 'val/labels/')
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Get all image filenames
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Shuffle and split into train and val
random.shuffle(image_files)
split_index = int(len(image_files) * train_split)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# Copy files to train and val subdirectories
def copy_files(file_list, src_images, src_labels, dst_images, dst_labels):
    for file in file_list:
        # Copy image
        src_image_path = os.path.join(src_images, file)
        dst_image_path = os.path.join(dst_images, file)
        shutil.copy(src_image_path, dst_image_path)
        
        # Copy label (if exists)
        label_file = os.path.splitext(file)[0] + '.txt'
        src_label_path = os.path.join(src_labels, label_file)
        dst_label_path = os.path.join(dst_labels, label_file)
        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dst_label_path)

# Copy train files
copy_files(train_files, images_dir, labels_dir, train_images_dir, train_labels_dir)

# Copy val files
copy_files(val_files, images_dir, labels_dir, val_images_dir, val_labels_dir)

print("Dataset split complete!")
print(f"Training data: {len(train_files)} images")
print(f"Validation data: {len(val_files)} images")
