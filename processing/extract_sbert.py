import torch
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import os
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser("(S)BERT Embeddings")
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dest", type=str, required=True)
    parser.add_argument("--model", type=str, default='sbert')
    args = parser.parse_args()
    print("Arguments: ", args)

    src = args.src
    model_type = args.model
    dest = args.dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    df_src = pd.read_csv(src)
    key = 'narration'
    data = {key: [], 'embedding': []}

    if model_type == 'sbert':
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')


    ### For ZSL with epic kitchen actions
    df = pd.DataFrame(columns=[key] + [f'{i}' for i in range(768)])
    for i, row in tqdm(enumerate(df_src.iterrows())):
        row = row[1]
        narration = row["narration"]
        description = row["llama_description"]

        if model_type == 'sbert':
            class_embedding = model.encode(description)
        else:
            encoded_input = tokenizer(description, return_tensors='pt')
            class_embedding = model(**encoded_input).pooler_output
            class_embedding = class_embedding.detach().numpy()[0]

        df.loc[i] = [narration] + list(class_embedding)
    
    df.to_csv(dest, sep=",", index=False)