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
- OPTIONAL: If you only want to export a single video (or a few) use the API:
```
curl -X GET "http://localhost:8080/api/projects/{project_id}/export?ids[]={task_id}&exportType=JSON_MIN" \
     -H 'Authorization: Token YOUR_API_TOKEN' \
     --output 'annotated_video.json'
```
- git clone https://github.com/hidenorly/video_annotations_to_yolo.git
- python ls2yolo.py --json_path JSON_PATH --video_path ~/.local/share/label-studio/media/upload/PROJECT_ID -o DATA_SET_PATH
- OPTIONAL: play with the parameters and number of augmentations in augment.py, change the input and output path, then run it to synthetically create more pictures (with augmentations)
- set images_dir, labels_dir and output_dir in split_data.py and run it (output_dir should be inside datasets/
## Train the model
- modify data.yaml to the correct path (datasets/DATA_SET_PATH)
- modify train.py input model and if needed other parameters, then run it
## Use the model
- use main.py -> detect all images in a directory (input/)
- use main2.py -> detect a video (adjust video_path)

## Models
Model | Base | imgsz | box_loss | cls_loss | dfl_loss | Box(P | R |mAP50 | mAP50-95 | Images | Description of the Model|
|-----|------|-------|----------|----------|----------|-------|---|------|----------|--------|-------------|
radcog-0.1.4 | yolo8l       | 512  | ?     | ?     | ?      | ? | ? |~0.5   | ~0.2  | ~5000 | First try with bad annotations (red balls)
radcog-0.2.1 | yolo11s      | 512  | 1.583 | 1.334 | 0.9287 | ? | ? | 0.617 | 0.345 | ~8000 | Second try with 2600 high quality black balls and images from radcog-0.1
radcog-0.2.2 | yolo11s      | 1028 | 1.449 | 1.576 | 1.245  | ? | ? | 0.657 | 0.414 | ~8000 | Upping quality with higher res and other small adjustments
radcog-0.2.3 | radcog-0.2.2 | 1028 | 1.415 | 1.346 | 1.174  | ? | ? | 0.68  | 0.426 | ~8000 | more epoch waiting, as 0.2.2 doesnt seemed finished with patience=5
radcog-0.3.1 | yolo11s | 1028 | ? | ? | ? | ? | ? | ? | ? | 2620 | only use perfect black ball annotations
radcog-0.4.2 | yolo11s | 1028 | 1.506 | 1.307 | 1.054 | 0.851 | 0.697 | 0.825 | 0.511 | 22900 | use 1.0 augmentations (x8) on the 2620 pictures and spin down the low learning rate etc. from 0.3. Not finished training but stopped after realising the training data was bad. Does not recognise black balls at all, despite "good" values
radcog-0.5.1 | yolo11s | 1028 | 1.882 | 1.212 | 0.9744 | 1 | 0.787 | 0.892 | 0.448 | 120 | realized all conversions were completely wrong making most of the annotated pictured bad. Fixed the conversion and reannotated around ~100 pictures for a first try (no augmentations). Instantly better than every model so far
