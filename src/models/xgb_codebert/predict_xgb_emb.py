# src/models/xgb_codebert/predict_xgb_emb.py

import os
import json
import joblib
import pandas as pd
import xgboost as xgb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from typing import List
from src.features.python_code_features import python_code_features

MODEL="models/xgb_codebert/xgb_codebert.json"
LABEL_ENCODER="models/xgb_codebert/xgb_codebert_label_encoder.pkl"
SCALER="models/xgb_codebert/xgb_codebert_scaler.pkl"
DENSE_FEATURE_NAMES="models/xgb_codebert/xgb_codebert_dense_feature_names.json"
EMBED_MODEL="microsoft/codebert-base"
INPUT="data/raw/test.csv"
OUTPUT="output/xgb_codebert_predictions.csv"
THRESHOLD=0.5

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
    # Ensure output directory exists
    out_dir = os.path.dirname(OUTPUT)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # Load artifacts
    le = joblib.load(LABEL_ENCODER)
    scaler = joblib.load(SCALER)
    dense_cols = load_dense_names(DENSE_FEATURE_NAMES)
    sk_model, booster = load_model_any(MODEL)

    # Initialize embedding model
    embed_model = SentenceTransformer(EMBED_MODEL)

    # Read input CSV
    df = pd.read_csv(INPUT)
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
    pred_enc = (proba >= THRESHOLD).astype(int)
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

    out_df.to_csv(OUTPUT, index=False)
    print(f"[OK] Wrote predictions -> {OUTPUT}")
    print(f"[Shapes] emb: {embeddings.shape}, dense: {F.shape}, stacked: {X.shape}")

if __name__ == '__main__':
    main()
