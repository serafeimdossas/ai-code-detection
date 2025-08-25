# src/models/mlp_codebert/predict_mlp.py

import os
import torch
import numpy as np
import joblib
import pandas as pd
from train_mlp_emb import MLP

MODEL = "models/mlp_codebert/mlp_emb.pt"
ENCODER = "models/mlp_codebert/mlp_emb_label_encoder.pkl"
EMBEDDINGS = "data/processed/codebert/test_emb.npy"
LABELS = "data/processed/codebert/test_labels.npy"
OUTPUT = "output/mlp_codebert_predictions.csv"

def main():
    # Ensure output directory exists
    out_dir = os.path.dirname(OUTPUT)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # Load
    state_dict = torch.load(MODEL)
    le         = joblib.load(ENCODER)
    
    # Build model
    # You need to know the embedding dim; e.g. load one embedding and inspect shape
    emb0 = np.load(EMBEDDINGS, allow_pickle=True)
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
    if LABELS:
        df["true"] = np.load(LABELS, allow_pickle=True)
        df["correct"] = np.where(df['pred'] == df['true'], 'CorrectPred', 'IncorrectPred')
    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT}")

if __name__=="__main__":
    main()
