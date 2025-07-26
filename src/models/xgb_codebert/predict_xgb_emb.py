# src/models/xgb_codebert/predict_xgb_emb.py

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
        "--model", type=str, default="models/xgb_codebert/xgb_with_emb.json",
        help="Path to the trained XGBoost model file"
    )
    parser.add_argument(
        "--encoder", type=str, default="models/xgb_codebert/xgb_with_emb_label_encoder.pkl",
        help="Path to the LabelEncoder pickle"
    )
    parser.add_argument(
        "--embed_model", type=str, default="microsoft/codebert-base",
        help="SentenceTransformer model name or path for embeddings"
    )
    parser.add_argument(
        "--input", type=str, default="data/raw/H-AIRosettaMP/test.csv",
        help="Path to a CSV file with a 'code' column containing snippets to predict"
    )
    parser.add_argument(
        "--output", type=str, default="output/xgb_codebert_predictions.csv",
        help="Path to write predictions CSV"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
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
    
    # Include ground truth if available and flag correctness
    if 'target' in df.columns:
        df['true_label'] = df['target']
        df['correct'] = np.where(df['pred_label'] == df['true_label'], 'CorrectPred', 'IncorrectPred')

    # Save
    df[['code', 'pred_label', 'prob_human'] + ([ 'true_label', 'correct'] if 'target' in df.columns else [])].to_csv(args.output, index=False)
    print(f"Wrote predictions to {args.output}")

if __name__ == '__main__':
    main()
