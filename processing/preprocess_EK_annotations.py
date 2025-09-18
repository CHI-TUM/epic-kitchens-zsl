import os
import argparse
import pandas as pd


def get_unique_narrations(input_dir: str, output_csv: str):
    """Load train+validation annotations, deduplicate by narration, and save to CSV."""

    filenames = ["EPIC_100_train.csv", "EPIC_100_validation.csv"]
    train_df = pd.read_csv(os.path.join(input_dir, filenames[0]))
    print("Train shape:", train_df.shape)
    dev_df = pd.read_csv(os.path.join(input_dir, filenames[1]))
    print("Validation shape:", dev_df.shape)

    all_df = pd.concat([train_df, dev_df], ignore_index=True)
    print("Combined shape:", all_df.shape)

    temp = all_df.groupby("narration").first().reset_index()
    final_df = temp[
        ["narration", "verb", "verb_class", "noun", "noun_class", "all_nouns", "all_noun_classes"]
    ]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, sep=",", index=False)
    print(f"Saved unique narrations to {output_csv}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract unique narrations from EpicKitchens annotations.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing EPIC_100_train.csv and EPIC_100_validation.csv",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save the deduplicated narrations CSV",
    )
    args = parser.parse_args()

    get_unique_narrations(args.input_dir, args.output_csv)

