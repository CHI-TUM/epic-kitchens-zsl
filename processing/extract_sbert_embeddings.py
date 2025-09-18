import torch
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import cdist
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


### BERT
def generate_bert_embeddings(path: str, output_path: str):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    df_src = pd.read_csv(path)
    key = 'narration'
    df = pd.DataFrame(columns=[key] + [f'{i}' for i in range(768)])

    for i, row in tqdm(enumerate(df_src.iterrows())):
        row = row[1]
        narration = row["narration"]
        description = row["llama_description"]

        encoded_input = tokenizer(description, return_tensors='pt')
        class_embedding = model(**encoded_input).pooler_output
        class_embedding = class_embedding.detach().numpy()[0]
        
        df.loc[i] = [narration] + list(class_embedding)
    df.to_csv(output_path, index=False)
    print(f"Saved BERT embeddings to {output_path}.")


### SBERT
def generate_sbert_embeddings(path: str, output_path: str):
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    df_src = pd.read_csv(path)
    key = 'narration'
    df = pd.DataFrame(columns=[key] + [f'{i}' for i in range(768)])

    for i, row in tqdm(enumerate(df_src.iterrows())):
        row = row[1]
        narration = row["narration"]
        description = row["llama_description"]

        class_embedding = model.encode(description)
        df.loc[i] = [narration] + list(class_embedding)

    df.to_csv(output_path, index=None)
    print(f"Saved BERT embeddings to {output_path}.")



if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="Generate BERT or SBERT embeddings."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to input CSV with narration and llama_description columns"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save embeddings CSV"
    )
    parser.add_argument(
        "--model",
        choices=["bert", "sbert"],
        default="bert",
        help="Which model to use for embeddings (default: bert)"
    )
    args = parser.parse_args()

    if args.model == "bert":
        generate_bert_embeddings(args.path, args.output_csv)
    else:
        generate_sbert_embeddings(args.path, args.output_csv)
