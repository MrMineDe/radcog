# radcog
radball ai recognition of the objects in a radball field/game

# Requirements
## Label Studio
used for labeling data, not strictly needed though
## Other Packages
These Packagenames are for arch-linux(some AUR), but should be similar for others
- python-gitpython
- python-matplotlib
- python-numpy
- python-opencv
- python-pillow
- python-psutil
- python-yaml
- python-requests
- python-scipy
- python-thop
- python-torchvision-cuda
- python-pytorch-cuda
- python-tqdm
- typhon-ultralytics

# Usage
## Label Data
- install label studio
- import videos (store those video files somewhere, where you can access them later)
- mark them (dont forget to check if you disabled automarking, when you only labeled part of a video)
## convert Label Studio to YOLO
- Export in JSON-MIN format
- git clone https://github.com/hidenorly/video_annotations_to_yolo.git
- python ls2yolo.py --json_path JSON_PATH --video_path ~/.local/share/label-studio/media/upload/2 -o DATA_SET_PATH
- set images_dir, labels_dir and output_dir in split_data.py and run it (output_dir should be inside datasets/
## Train the model
- modify data.yaml to the correct path (datasets/DATA_SET_PATH)
- modify train.py input model and if needed other parameters, then run it
## Use the model
- use main.py -> detect all images in a directory (input/)
- use main2.py -> detect a video (adjust video_path)

## Models
Model | Base | imgsz | box_loss | cls_loss | dfl_loss | mAP50 | mAP50-95 | Images | Description |
|-----|------|-------|----------|----------|----------|-------|----------|--------|-------------|
radcog-0.1.4 | yolo8l       | 512  | ?     | ?     | ?      |~0.5   | ~0.2  | ~5000 | First try with bad annotations (red balls)
radcog-0.2.1 | yolo11s      | 512  | 1.583 | 1.334 | 0.9287 | 0.617 | 0.345 | ~8000 | Second try with 2600 high quality black balls and images from radcog-0.1
radcog-0.2.2 | yolo11s      | 1028 | 1.449 | 1.576 | 1.245  | 0.657 | 0.414 | ~8000 | Upping quality with higher res and other small adjustments
radcog-0.2.3 | radcog-0.2.2 | 1028 | ? | ? | ? | ? | ? | ~8000 | more epoch waiting, as 0.2.2 doesnt seemed finished with patience=5
