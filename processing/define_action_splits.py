import os
import random
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Define splits.")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory of EpicKitchens (contains participant folders with videos)"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Directory containing EPIC_100_train.csv and EPIC_100_validation.csv"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save split CSVs"
    )
    args = parser.parse_args()

    root = args.root
    path = args.annotations
    splits_dest = args.output

    os.makedirs(splits_dest, exist_ok=True)
    filenames = ["EPIC_100_train.csv", "EPIC_100_validation.csv"]

    train_df = pd.read_csv(os.path.join(path, filenames[0]))
    print(train_df.shape)
    dev_df = pd.read_csv(os.path.join(path, filenames[1]))
    print(dev_df.shape)

    all_df = pd.concat([train_df, dev_df], ignore_index=True)
    all_df.head(5)

    grouped = train_df.groupby(['verb_class', 'all_noun_classes'])
    grouped.head(5)

    group_keys = list(grouped.groups.keys())

    random.shuffle(group_keys)

    train_keys, dev_test_keys = train_test_split(group_keys, test_size=0.3, random_state=42)
    dev_keys, test_keys = train_test_split(group_keys, test_size=0.5, random_state=42)
    len(train_keys), len(dev_keys), len(test_keys)

    train_df = pd.concat([grouped.get_group(key) for key in train_keys])
    dev_df = pd.concat([grouped.get_group(key) for key in dev_keys])
    test_df = pd.concat([grouped.get_group(key) for key in test_keys])


    train_df.to_csv(os.path.join(splits_dest, 'train.csv'), sep=',', index=False)
    dev_df.to_csv(os.path.join(splits_dest, 'devel.csv'), sep=',', index=False)
    test_df.to_csv(os.path.join(splits_dest, 'test.csv'), sep=',', index=False)


    ### -------------------------- ###

    
    # Again, only with existing audio files
    # Define the splits only for the existing audio files
    train_df = pd.read_csv(os.path.join(path, filenames[0]))
    print(train_df.shape)


    valid_rows = []
    for row in tqdm(train_df.iterrows()):
        row = row[1]
        participant_id = row['participant_id']
        video_id = row['video_id']
        start_time = row['start_timestamp'] 
        end_time = row['stop_timestamp'] 
        
        input_fn = os.path.join(root, participant_id, 'videos', f'{video_id}.MP4')
        if os.path.isfile(input_fn):
            valid_rows.append(row)

    filtered_df = pd.DataFrame(valid_rows)


    grouped_exist = filtered_df.groupby(['verb_class', 'all_noun_classes'])
    group_keys = list(grouped_exist.groups.keys())
    random.shuffle(group_keys)

    train_keys, dev_test_keys = train_test_split(group_keys, test_size=0.3, random_state=42)
    dev_keys, test_keys = train_test_split(group_keys, test_size=0.5, random_state=42)
    len(train_keys), len(dev_keys), len(test_keys)


    train_df = pd.concat([grouped_exist.get_group(key) for key in train_keys])
    dev_df = pd.concat([grouped_exist.get_group(key) for key in dev_keys])
    test_df = pd.concat([grouped_exist.get_group(key) for key in test_keys])
    print(train_df.shape), print(dev_df.shape), print(test_df.shape)


    splits_dest = os.path.join(splits_dest, "splits/splits_available")
    os.makedirs(splits_dest, exist_ok=True)
    train_df.to_csv(os.path.join(splits_dest, 'train.csv'), sep=',', index=False)
    dev_df.to_csv(os.path.join(splits_dest, 'devel.csv'), sep=',', index=False)
    test_df.to_csv(os.path.join(splits_dest, 'test.csv'), sep=',', index=False)