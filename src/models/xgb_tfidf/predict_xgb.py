# src/models/xgb_tfidf/predict.py

import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_code(snippet: str) -> str:
    if not isinstance(snippet, str):
        return ""
    return " ".join(snippet.split())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict human vs. AI code snippets using trained XGBoost model"
    )
    parser.add_argument(
        "--model", type=str, default="models/xgb_tfidf/xgb_baseline.json",
        help="Path to the trained XGBoost model file"
    )
    parser.add_argument(
        "--vectorizer", type=str, default="data/processed/tfidf/tfidf_vectorizer.pkl",
        help="Path to the fitted TF-IDF vectorizer pickle"
    )
    parser.add_argument(
        "--label_encoder", type=str, default="models/xgb_tfidf/xgb_baseline_label_encoder.pkl",
        help="Path to the LabelEncoder pickle"
    )
    parser.add_argument(
        "--input", type=str, default="data/raw/H-AIRosettaMP/test.csv",
        help="Path to a CSV file with a column 'code' containing snippets to predict"
    )
    parser.add_argument(
        "--output", type=str, default="output/xgb_tfidf_predictions.csv",
        help="Path to write predictions CSV"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # Load artifacts
    vect = joblib.load(args.vectorizer)
    model = joblib.load(args.model) if args.model.endswith('.pkl') else None
    
    # if model is None:
    # use xgboost native load
    import xgboost as xgb
    bst = xgb.Booster()
    bst.load_model(args.model)
    
    le = joblib.load(args.label_encoder)

    # Read input CSV
    df = pd.read_csv(args.input)
    if 'code' not in df.columns:
        raise KeyError("Input CSV must contain a 'code' column")
    
    # Clean
    df['code_clean'] = df['code'].apply(clean_code)
    
    # Vectorize
    X = vect.transform(df['code_clean'])

    # Predict
    if model:
        probas = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
    else:
        dmat = xgb.DMatrix(X)
        probas = bst.predict(dmat)
        preds = (probas >= 0.5).astype(int)

    # Map back to labels
    df['pred_encoded'] = preds
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
