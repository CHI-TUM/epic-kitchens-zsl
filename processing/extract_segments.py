import os
import argparse
import sys
import subprocess
import csv
import pandas as pd
from tqdm import tqdm

def extract_segment(input_file, output_dir, start_time, end_time, video_id):
    
    start_time = start_time
    end_time = end_time
    # output_file = f"{output_dir}/output_{start_time}_{end_time}.mp4"
    output_file = f"{output_dir}/audio_{video_id}_{start_time}_{end_time}.mp3"
    output_file = output_file.replace(":", "-")
    if os.path.isfile(output_file):
        return None

    # Run FFmpeg command using subprocess
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-ss', start_time,
        '-to', end_time,
        '-vn', # disable video recording
        # '-c', 'copy',
        '-acodec', 'mp3',
        output_file
    ]
    print("command: ", cmd)
    subprocess.run(cmd)

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract segments."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory of the EpicKitchens dataset, containing the annotations"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory where extracted segments will be stored"
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "both"],
        default="validation",
        help="Which split to process (default: validation)"
    )
    args = parser.parse_args()

    root = args.root
    output_dir = args.outdir

    annotations_dir = os.path.join(root, "epic-kitchens-100-annotations")
    os.makedirs(output_dir, exist_ok=True)

    filenames = ["EPIC_100_train.csv", "EPIC_100_validation.csv"]
    train_df = pd.read_csv(os.path.join(annotations_dir, filenames[0]))
    dev_df = pd.read_csv(os.path.join(annotations_dir, filenames[1]))
    all_df = pd.concat([train_df, dev_df], ignore_index=True)

    if args.split == "train":
        df = train_df
        print("train_df:", df.shape)
    elif args.split == "validation":
        df = dev_df
        print("dev_df:", df.shape)
    else: 
        df = all_df
        print("all_df:", df.shape)
    print("dev_df: ", dev_df.shape)

    output_rows = []
    counter = 0
    for row in tqdm(df.iterrows()):
        row = row[1]
        participant_id = row['participant_id']
        video_id = row['video_id']
        start_time = row['start_timestamp'] 
        end_time = row['stop_timestamp'] 
        
        input_fn = os.path.join(root, participant_id, 'videos', f'{video_id}.MP4')
        if not os.path.isfile(input_fn):
            counter += 1 
            continue
        output_file = extract_segment(input_fn, output_dir, start_time, end_time, video_id)
        if output_file is None:
            continue

        row['audio_filename'] = os.path.relpath(output_file, output_dir)
        output_rows.append(row)

    print("Counter: ", counter)
    df_new = pd.DataFrame(output_rows)
    df_new.to_csv(os.path.join(output_dir, 'audio_segments.csv'), sep=',', index=False)
