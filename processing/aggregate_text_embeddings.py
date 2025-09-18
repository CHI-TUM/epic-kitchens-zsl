import os
import argparse
import pandas as pd


def average_embeddings(overview_csv, embeddings_csv, output_csv, target_column):
    # Load overview (contains narrations + targets)
    overview_df = pd.read_csv(overview_csv)
    print("Overview:", overview_df.shape)
    print(overview_df.head(5))

    # Load embeddings (indexed by narration)
    embeddings_df = pd.read_csv(embeddings_csv)
    embeddings_df.set_index("narration", inplace=True)
    print("Embeddings:", embeddings_df.shape)
    print(embeddings_df.head(5))

    # Group narrations by target (e.g., verb_class)
    grouped = overview_df.groupby(target_column)

    result_dict = {}
    for target, group in grouped:
        target_rows = embeddings_df.loc[group["narration"]]
        mean_values = target_rows.mean(axis=0)
        result_dict[target] = mean_values

    # Convert to DataFrame
    result_df = pd.DataFrame(result_dict).transpose()
    result_df.reset_index(inplace=True)
    result_df.rename(columns={"index": target_column}, inplace=True)

    print("\nResult:", result_df.shape)
    print(result_df.head(5))

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_df.to_csv(output_csv, sep=",", index=False)
    print(f"Saved averaged embeddings to {output_csv}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute averaged text embeddings."
    )
    parser.add_argument(
        "--overview_csv",
        type=str,
        required=True,
        help="CSV file with audio segments and labels (contains 'narration' column).",
    )
    parser.add_argument(
        "--embeddings_csv",
        type=str,
        required=True,
        help="CSV file with embeddings (indexed by narration).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save averaged embeddings CSV.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="verb_class",
        help="Column to group by when averaging embeddings (default: verb_class).",
    )
    args = parser.parse_args()

    average_embeddings(
        args.overview_csv,
        args.embeddings_csv,
        args.output_csv,
        args.target_column,
    )


# overview_df = pd.read_csv("/nas/staff/data_work/AG/Audionomous/EpicKitchens/extracted_segments/train/audio_segments.csv")
# embeddings_df = pd.read_csv("/nas/staff/data_work/AG/Audionomous/EpicKitchens/bert-embeddings/sbert_embeddings.csv")
# embeddings_df.set_index('narration', inplace=True)
# print(embeddings_df.head(5))


# target_column = 'verb_class'
# grouped = overview_df.groupby(target_column)

# result_dict = {}
# for target, group in grouped:
#     target_rows = embeddings_df.loc[group['narration']]
#     mean_values = target_rows.mean(axis=0)
#     result_dict[target] = mean_values

# result_df = pd.DataFrame(result_dict).transpose()
# result_df.reset_index(inplace=True)
# result_df.rename(columns={'index': target_column}, inplace=True)
# print("\nResult df: ", result_df)
# result_df.to_csv(f"/nas/staff/data_work/AG/Audionomous/EpicKitchens/averaged_text_embeddings/{target_column}_sbert.csv", sep=',', index=False)