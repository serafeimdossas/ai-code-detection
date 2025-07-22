# src/models/xgb_emb/predict_xgb_emb.py

import os
import argparse
import joblib
import pandas as pd
import xgboost as xgb
import numpy as np
from sentence_transformers import SentenceTransformer


def clean_code(snippet: str) -> str:
    if not isinstance(snippet, str):
        return ""
    return " ".join(snippet.split())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict human vs. AI code snippets using embedding-based XGBoost model"
    )
    parser.add_argument(
        "--model", type=str, default="models/xgb_emb/xgb_with_emb.json",
        help="Path to the trained XGBoost model file"
    )
    parser.add_argument(
        "--encoder", type=str, default="models/xgb_emb/xgb_with_emb_label_encoder.pkl",
        help="Path to the LabelEncoder pickle"
    )
    parser.add_argument(
        "--embed_model", type=str, default="microsoft/codebert-base",
        help="SentenceTransformer model name or path for embeddings"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to a CSV file with a 'code' column containing snippets to predict"
    )
    parser.add_argument(
        "--output", type=str, default="predictions.csv",
        help="Path to write predictions CSV"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Load label encoder and model
    le = joblib.load(args.encoder)
    bst = xgb.Booster()
    bst.load_model(args.model)

    # Initialize embedding model
    embed_model = SentenceTransformer(args.embed_model)

    # Read input CSV
    df = pd.read_csv(args.input)
    if 'code' not in df.columns:
        raise KeyError("Input CSV must contain a 'code' column")

    # Clean
    df['code_clean'] = df['code'].apply(clean_code)

    # Generate embeddings for all snippets
    embeddings = embed_model.encode(
        df['code_clean'].tolist(),
        batch_size=32,
        show_progress_bar=True
    )
    X = np.array(embeddings)

    # Predict
    dmat = xgb.DMatrix(X)
    probas = bst.predict(dmat)
    preds = (probas >= 0.5).astype(int)

    # Map back to labels
    df['pred_label'] = le.inverse_transform(preds)
    df['prob_human'] = probas

    # Save
    df[['code', 'pred_label', 'prob_human']].to_csv(args.output, index=False)
    print(f"Wrote predictions to {args.output}")

if __name__ == '__main__':
    main()
