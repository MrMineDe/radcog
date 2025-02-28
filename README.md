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
- python-ultralytics
- python-lap
- python-opencv

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
Model | Base | imgsz | box_loss | cls_loss | dfl_loss | Box(P | R |mAP50 | mAP50-95 | Images | Description |
|-----|------|-------|----------|----------|----------|-------|---|------|----------|--------|-------------------------------|
radcog-0.5.4 | yolo11s | 1028 | ? | ? | ? | ? | ? | ? | ? | 454 | more annotation data than 0.5.1, no augmentations
radcog-0.5.5 | yolo11s | 1028 | 0.7568 | 0.5142 | 0.8323 | 0.994 | 0.926 | 0.981 | 0.826 | 2025 | more annotations than 0.5.4, no augmentations
<table border="1">
    <tr>
        <th>Model</th>
        <th>Base</th>
        <th>imgsz</th>
        <th>box_loss</th>
        <th>cls_loss</th>
        <th>dfl_loss</th>
        <th>Box(P)</th>
        <th>R</th>
        <th>mAP50</th>
        <th>mAP50-95</th>
        <th>Images</th>
    </tr>
    <tr>
        <td colspan="11"><strong>First try with bad annotations (red balls)</strong></td>
    </tr>
    <tr>
        <td>radcog-0.1.4</td>
        <td>yolo8l</td>
        <td>512</td>
        <td>?</td>
        <td>?</td>
        <td>?</td>
        <td>?</td>
        <td>?</td>
        <td>~0.5</td>
        <td>~0.2</td>
        <td>~5000</td>
    </tr>
    <tr>
        <td colspan="11"><strong>Second try with 2600 high quality black balls and images from radcog-0.1</strong></td>
    </tr>
    <tr>
        <td>radcog-0.2.1</td>
        <td>yolo11s</td>
        <td>512</td>
        <td>1.583</td>
        <td>1.334</td>
        <td>0.9287</td>
        <td>?</td>
        <td>?</td>
        <td>0.617</td>
        <td>0.345</td>
        <td>~8000</td>
    </tr>
    <tr>
        <td colspan="11"><strong>Upping quality with higher res and other small adjustments</strong></td>
    </tr>
    <tr>
        <td>radcog-0.2.2</td>
        <td>yolo11s</td>
        <td>1028</td>
        <td>1.449</td>
        <td>1.576</td>
        <td>1.245</td>
        <td>?</td>
        <td>?</td>
        <td>0.657</td>
        <td>0.414</td>
        <td>~8000</td>
    </tr>
    <tr>
        <td colspan="11"><strong>More epoch waiting, as 0.2.2 doesn’t seem finished with patience=5</strong></td>
    </tr>
    <tr>
        <td>radcog-0.2.3</td>
        <td>radcog-0.2.2</td>
        <td>1028</td>
        <td>1.415</td>
        <td>1.346</td>
        <td>1.174</td>
        <td>?</td>
        <td>?</td>
        <td>0.68</td>
        <td>0.426</td>
        <td>~8000</td>
    </tr>
    <tr>
        <td colspan="11"><strong>Only use perfect black ball annotations</strong></td>
    </tr>
    <tr>
        <td>radcog-0.3.1</td>
        <td>yolo11s</td>
        <td>1028</td>
        <td>?</td>
        <td>?</td>
        <td>?</td>
        <td>?</td>
        <td>?</td>
        <td>?</td>
        <td>?</td>
        <td>2620</td>
    </tr>
    <tr>
        <td colspan="11"><strong>Use 1.0 augmentations (x8) on the 2620 pictures and spin down the low learning rate etc. from 0.3. Not finished training but stopped after realizing the training data was bad. Does not recognize black balls at all, despite "good" values.</strong></td>
    </tr>
    <tr>
        <td>radcog-0.4.2</td>
        <td>yolo11s</td>
        <td>1028</td>
        <td>1.506</td>
        <td>1.307</td>
        <td>1.054</td>
        <td>0.851</td>
        <td>0.697</td>
        <td>0.825</td>
        <td>0.511</td>
        <td>22900</td>
    </tr>
    <tr>
        <td colspan="11"><strong>Realized all conversions were completely wrong, making most of the annotated pictures bad. Fixed the conversion and reannotated around ~100 pictures for a first try (no augmentations). Instantly better than every model so far.</strong></td>
    </tr>
    <tr>
        <td>radcog-0.5.1</td>
        <td>yolo11s</td>
        <td>1028</td>
        <td>1.882</td>
        <td>1.212</td>
        <td>0.9744</td>
        <td>1</td>
        <td>0.787</td>
        <td>0.892</td>
        <td>0.448</td>
        <td>120</td>
    </tr>
    <tr>
        <td colspan="11"><strong>Use radcog-0.5.1 data with more conservative augmentations (x8) than in 0.4.2.</strong></td>
    </tr>
    <tr>
        <td>radcog-0.5.3</td>
        <td>yolo11s</td>
        <td>1028</td>
        <td>1.544</td>
        <td>0.8294</td>
        <td>1.026</td>
        <td>0.918</td>
        <td>0.777</td>
        <td>0.856</td>
        <td>0.462</td>
        <td>~1000</td>
    </tr>
</table>
