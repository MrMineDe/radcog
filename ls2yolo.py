"""
This script processes video annotations created using Label Studio, converting them into a
format suitable for the YOLO object detection model. The script has the capability to interpolate
bounding boxes for each intermediate frame based on key-frame annotations (as needed), and export
these labels (i.e., bounding box coordinates), along with the corresponding frames, into a
YOLO-compatible format. As it stands with Label Studio version 1.7.0, such functionality isn't
inherently available. Please note that video annotations should be exported in the JSON-MIN format.

The script was originally written by github.com/hidenorly and github.com/deepinvalue.
I have altered it a bit, mainly added the fps and changing from "time" to "frame"
For the copy I modified see: https://github.com/hidenorly/video_annotations_to_yolo/tree/SupportMultipleVideoAndOutputOnlyNecessaryFrames
"""

import argparse
import os
import json
import csv
import copy
import cv2
from decimal import Decimal
from pathlib import Path
from tqdm import tqdm

def linear_interpolation(prev_seq, seq, label):
    # Define the start and end frame numbers
    a0 = prev_seq['frame']
    a1 = seq['frame']
    frames_info = dict()
    # Loop over all intermediate frames
    for frame in range(a0+1, a1):
        t = Decimal(frame-a0)/Decimal(a1-a0)
        info = [label]
        # Interpolate bounding box dimensions for the current frame
        for b0, b1 in ((prev_seq[k], seq[k]) for k in ('x', 'y', 'width', 'height')):
            info.append(str(b0 + t*(b1-b0)))
        # Add interpolated information for the current frame to 'frames_info'
        frames_info[frame] = info
    return frames_info

def main(video_label, output_base, offset=0, labels=set(), custom_fps=None):
    video_path = video_label["video"]
    labels_dict = {k:i for i,k in enumerate(labels)}

    # Open the video to get frame rate and total frames
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    video_fps = vidcap.get(cv2.CAP_PROP_FPS)  # Get video frame rate
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frames
    vidcap.release()

    # Use custom FPS if provided, otherwise use video FPS
    fps = custom_fps if custom_fps is not None else video_fps
    print(f"Video FPS: {video_fps}, Custom FPS: {fps}, Total Frames: {total_frames}")

    # extract count of frames
    max_frames = 0
    for subject in video_label['box']:
        if "framesCount" in subject:
            _frame_count = int( subject["framesCount"] )
            if _frame_count > max_frames:
                max_frames = _frame_count

    # prepare metadata available frame flags
    availableFrames = [False] * (max_frames+1)
    for subject in video_label['box']:
        start_frame = subject['sequence'][0]["frame"] - 1  # Shift frame numbers by -1
        end_frame = subject['sequence'][len(subject['sequence'])-1]["frame"] - 1  # Shift frame numbers by -1
        print(f'{subject['labels'][0]} : {start_frame+offset} - {end_frame+offset}')
        for i in range(start_frame, end_frame):
            availableFrames[i] = True

    # Initialize dictionaries to store file information
    files_dict = dict()

    # Loop over the subjects, i.e. football players in a match
    for subject in copy.deepcopy(video_label['box']):
        # Get the subject labels (e.g. team-A, team-B, referee, ball)
        subject_labels = subject['labels']

        # Map the label to its integer representation
        if len(subject_labels)==1:
            label = labels_dict[subject_labels[0]]
        else:
            raise ValueError("Each subject must have exactly one label.")

        prev_seq = None

        # Process each sequence in the subject's timeline
        for seq in subject['sequence']:
            frame = seq['frame'] + 1  # Shift frame numbers by -1

            # Adjust the x and y coordinates to be the center of the bounding box
            seq['x'] += seq['width'] / Decimal('2')
            seq['y'] += seq['height'] / Decimal('2')

            # Normalize bounding box dimensions (Label Studio JSON uses percentages)
            for k in ('x', 'y', 'width', 'height'):
                seq[k] /= Decimal('100')

            # If the current sequence is not adjacent to the previous sequence, perform linear interpolation
            if (prev_seq is not None) and prev_seq['enabled'] and (frame - prev_seq['frame'] > 1):
                lines = linear_interpolation(prev_seq, seq, label)
            else:
                lines = dict()

            # Create the bounding box information line for the current frame
            lines[frame] = [label] + [str(seq[k]) for k in ('x', 'y', 'width', 'height')]

            # Add the bounding box information line to the corresponding frame in 'files_dict'
            for frame, info in lines.items():
                if frame in files_dict:
                    files_dict[frame].append(info)
                else:
                    files_dict[frame] = [info]

            prev_seq = seq

    # Sort the file information dictionary
    files_dict = dict(sorted(files_dict.items()))

    print("Exporting annotations in YOLO format")

    # Prepare YOLO directory structure
    output_path = Path(output_base)
    [(output_path / p).mkdir(parents=True, exist_ok=True) for p in ('images/', 'labels/')]

    max_frame = max(files_dict.keys())
    padding = len(str(max_frame))

    # Write the YOLO labels
    for frame, lines in files_dict.items():
        if availableFrames[frame]:
            with open(output_path / 'labels' / f'frame_{offset+frame:0{padding}d}.txt', 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ')
                csvwriter.writerows(lines)

    # Extract the Frames
    if os.path.isfile(video_path):
        vidcap = cv2.VideoCapture(video_path)
        print(f'Extracting frames')
        for frame in tqdm(files_dict):
            if availableFrames[frame]:
                # Calculate the timestamp for the frame using custom FPS
                timestamp = frame / fps  # Convert frame number to timestamp
                vidcap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)  # Set timestamp in milliseconds
                success, image = vidcap.read()
                if success:
                    cv2.imwrite(str(output_path / 'images' / f'frame_{offset+frame:0{padding}d}.jpg'), image)
                else:
                    print(f"Unable to read frame {frame}. Quiting.")
                    break

    print("Process finished successfully.")

    return offset+len(tqdm(files_dict))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""This script processes video annotations exported from Label Studio
        in JSON-MIN format, converting them into a YOLO-compatible format. The script supports
        interpolation of bounding boxes for intermediate frames based on key-frame annotations
        and exports these labels along with corresponding frames (if a video path is provided).""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("-j", "--json_path", required=True, help="Path to JSON annotations")
    parser.add_argument("-v", "--video_path", default=None, help="Optional path to video file."\
        " If provided, corresponding frames will be extracted.")
    parser.add_argument("-o", "--output_base", default='output/', help="Path to output base directory")
    parser.add_argument("--fps", type=float, default=None, help="Custom frame rate (FPS) for frame extraction.")
    args = parser.parse_args()

    print("Parsing annotations from JSON")
    video_labels = []
    with open(args.json_path) as f:
        video_labels = json.load(f, parse_float=Decimal)
        # replace from video uri in the export file to actual path
        for video in video_labels:
            if "video" in video:
                video_path = video["video"]
                pos = video_path.rfind("/")
                if pos!=None:
                    video_path = os.path.join(args.video_path, video_path[pos+1:])
                    if os.path.isfile(video_path):
                        video["video"] = video_path

    # extract labels
    labels = set()
    for video_label in video_labels:
        for subject in video_label['box']:
            labels.add(*subject['labels'])
    labels = sorted(labels)
    print(f'labels = {labels}')

    # extract video frames and output yolo formated frame_xxxx.txt
    frame_count = 0
    for video_label in video_labels:
        print(f'{video_label["video"]}......')
        frame_count += main(video_label, args.output_base, frame_count, labels, custom_fps=args.fps)

    # Write the YOLO classes
    with open(os.path.join(args.output_base, f'classes.txt'), 'w') as f:
        f.writelines(f'{line}\n' for line in labels)
