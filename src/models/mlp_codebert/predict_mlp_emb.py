# src/models/mlp_codebert/predict_mlp.py

import os
import argparse
import torch
import numpy as np
import joblib
import pandas as pd
from train_mlp_emb import MLP 

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict human vs. AI code snippets using embedding-based MLP model"
    )
    parser.add_argument(
        "--model", type=str, default="models/mlp_codebert/mlp_emb.pt",
        help="Path to the trained MLP model file"
    )
    parser.add_argument(
        "--encoder", type=str, default="models/mlp_codebert/mlp_emb_label_encoder.pkl",
        help="Path to the LabelEncoder pickle"
    )
    parser.add_argument(
        "--embeddings", type=str, default="data/processed/codebert/test_emb.npy",
        help="SentenceTransformer model name or path for embeddings"
    )
    parser.add_argument(
        "--labels", type=str, default="data/processed/codebert/test_labels.npy",
        help="Path to true labels."
    )
    parser.add_argument(
        "--out", type=str, default="output/mlp_codebert_predictions.csv",
        help="Path to write predictions CSV"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure output directory exists
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # Load
    state_dict = torch.load(args.model)
    le         = joblib.load(args.encoder)
    
    # Build model
    # You need to know the embedding dim; e.g. load one embedding and inspect shape
    emb0 = np.load(args.embeddings, allow_pickle=True)
    model = MLP(input_dim=emb0.shape[1])
    model.load_state_dict(state_dict)
    model.eval()

    # Run on the entire embedding file
    X = torch.from_numpy(emb0).float()
    with torch.no_grad():
        probs = model(X).squeeze().numpy()
    preds = (probs >= 0.5).astype(int)
    labels = le.inverse_transform(preds)

    # Save
    df = pd.DataFrame({"pred": labels, "prob_human": probs})
    if args.labels:
        df["true"] = np.load(args.labels, allow_pickle=True)
        df["correct"] = np.where(df['pred'] == df['true'], 'CorrectPred', 'IncorrectPred')
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

if __name__=="__main__":
    main()
