# src/models/xgb_codebert/predict_xgb_emb.py

import os
import json
import argparse
import joblib
import pandas as pd
import xgboost as xgb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from typing import List
from src.features.python_code_features import python_code_features

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict human vs. AI code snippets using embedding-based XGBoost model"
    )
    parser.add_argument(
        "--model", type=str, default="models/xgb_codebert/xgb_codebert.json",
        help="Path to the trained XGBoost model file"
    )
    parser.add_argument(
        "--label_encoder", type=str, default="models/xgb_codebert/xgb_codebert_label_encoder.pkl",
        help="Path to the LabelEncoder pickle"
    )
    parser.add_argument(
        "--scaler", type=str, default="models/xgb_codebert/xgb_codebert_scaler.pkl",
        help="StandardScaler pickle for dense features"
    )
    parser.add_argument(
        "--dense_feature_names", type=str, default="models/xgb_codebert/xgb_codebert_dense_feature_names.json",
        help="JSON list of dense feature columns in training order"
    )
    parser.add_argument("--embed_model", type=str, default="microsoft/codebert-base")
    parser.add_argument("--input", type=str, default="data/raw/test.csv")
    parser.add_argument("--output", type=str, default="output/xgb_codebert_predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()

def clean_for_embed(snippet: str) -> str:
    if not isinstance(snippet, str):
        return ""
    return snippet.replace("\r\n", "\n").replace("\r", "\n")

def clean_for_feats(snippet: str) -> str:
    if not isinstance(snippet, str):
        return ""
    s = snippet.replace("\r\n", "\n").replace("\r", "\n")
    # keep indentation; strip trailing spaces per line
    return "\n".join(line.rstrip() for line in s.split("\n"))

def load_dense_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_dense_df(code_series: pd.Series) -> pd.DataFrame:
    feats = code_series.fillna("").map(python_code_features).apply(pd.Series)
    return feats.astype("float32")

def align_dense(F: pd.DataFrame, ordered_cols: List[str]) -> pd.DataFrame:
    # add missing as zeros; drop extras; enforce order
    for c in ordered_cols:
        if c not in F.columns:
            F[c] = 0.0
    return F[ordered_cols]

def load_model_any(path: str):
    if path.endswith(".pkl"):
        return joblib.load(path), None
    booster = xgb.Booster()
    booster.load_model(path)
    return None, booster

def main():
    args = parse_args()
    
    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # Load artifacts
    le = joblib.load(args.label_encoder)
    scaler = joblib.load(args.scaler)
    dense_cols = load_dense_names(args.dense_feature_names)
    sk_model, booster = load_model_any(args.model)

    # Initialize embedding model
    embed_model = SentenceTransformer(args.embed_model)

    # Read input CSV
    df = pd.read_csv(args.input)
    if 'code' not in df.columns:
        raise KeyError("Input CSV must contain a 'code' column")

    # Clean
    df["code_embed"] = df["code"].apply(clean_for_embed)
    df["code_feats"] = df["code"].apply(clean_for_feats)

    # Generate embeddings for all snippets
    embeddings = embed_model.encode(
        df['code_embed'].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False # ???
    ).astype("float32")

    # engineer code features
    F = build_dense_df(df["code_feats"])
    F = align_dense(F, dense_cols)
    F_scaled = scaler.transform(F.values.astype(np.float32)) 

    # Stack
    X = np.hstack([embeddings, F_scaled]).astype("float32")

    # predict
    if sk_model is not None:
        proba = sk_model.predict_proba(X)[:, 1]
    else:
        dmat = xgb.DMatrix(X)
        proba = booster.predict(dmat) # type: ignore
    pred_enc = (proba >= args.threshold).astype(int)
    pred_label = le.inverse_transform(pred_enc)

    # Output
    out = {}
    if "task_name" in df.columns: out["task_name"] = df["task_name"]
    out_df = pd.DataFrame({
        **out,
        "pred_label": pred_label,
        "prob_positive": proba
    })

    # If labels present, add correctness
    if "label" in df.columns:
        out_df["true_label"] = df["label"]
        out_df["correct"] = (out_df["pred_label"] == out_df["true_label"]).map({True: "CorrectPred", False: "IncorrectPred"})

    out_df.to_csv(args.output, index=False)
    print(f"[OK] Wrote predictions -> {args.output}")
    print(f"[Shapes] emb: {embeddings.shape}, dense: {F.shape}, stacked: {X.shape}")

if __name__ == '__main__':
    main()
