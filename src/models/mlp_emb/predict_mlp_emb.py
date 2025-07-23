# src/models/mlp/predict_mlp.py

import argparse
import torch
import numpy as np
import joblib
import pandas as pd
from train_mlp_emb import MLP 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    required=True)  # e.g. models/mlp_emb/mlp_emb.pt
    p.add_argument("--encoder",  required=True)  # e.g. models/mlp_emb/mlp_emb_label_encoder.pkl
    p.add_argument("--embeddings", required=True) # e.g. data/processed/codebert/test_emb.npy
    p.add_argument("--labels",   required=False) # optional, for evaluation
    p.add_argument("--out",      default="mlp_predictions.csv")
    return p.parse_args()

def main():
    args = parse_args()
    
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
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

if __name__=="__main__":
    main()
