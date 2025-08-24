# src/models/lr_tfidf/predict_lr_tfidf.py

import os
import json
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix

# Import the feature extraction function
from src.features.python_code_features import python_code_features

MODEL="models/lr_tfidf/lr_tfidf.pkl"
VECTORIZER="data/processed/tfidf/tfidf_vectorizer.pkl"
LABEL_ENCODER="models/lr_tfidf/lr_tfidf_label_encoder.pkl"
SCALER="models/lr_tfidf/lr_tfidf_scaler.pkl"
DENSE_FEATURE_NAMES="models/lr_tfidf/lr_tfidf_dense_features.json"
SCALE_DENSE=True
INPUT="data/raw/test.csv"
OUTPUT="output/lr_tfidf_predictions.csv"

def clean_for_tfidf(snippet: str) -> str:
    # Clean code snippet for TF-IDF vectorization
    if not isinstance(snippet, str):
        return ""
    return " ".join(snippet.split())

def clean_for_feats(snippet: str) -> str:
    # Clean code snippet for feature extraction
    if not isinstance(snippet, str):
        return ""
    s = snippet.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in s.split("\n"))

# Build dense features DataFrame from code snippets
def build_dense_features_df(code_series: pd.Series) -> pd.DataFrame:
    feats = code_series.fillna("").map(python_code_features).apply(pd.Series)
    return feats.astype("float32")

# Align dense features with the training set
def align_dense_features(F: pd.DataFrame, names_path: str) -> pd.DataFrame:
    with open(names_path, "r", encoding="utf-8") as f:
        train_cols = json.load(f)
    # add missing -> zeros
    for c in train_cols:
        if c not in F.columns:
            F[c] = 0.0
    # keep order & drop extras
    F = F[train_cols]
    return F

# Ensure all matrices are in CSR format
def ensure_csr(*mats):
    from scipy.sparse import csr_matrix, issparse, isspmatrix_csr
    out = []
    for M in mats:
        if not issparse(M):
            M = csr_matrix(M)
        elif not isspmatrix_csr(M):
            M = M.tocsr()
        out.append(M)
    return out

def main():    
    # Ensure output directory exists
    out_dir = os.path.dirname(OUTPUT)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Load artifacts
    vect = joblib.load(VECTORIZER)
    le = joblib.load(LABEL_ENCODER)
    scaler = joblib.load(SCALER)
    sk_model = joblib.load(MODEL)

    # Input
    df = pd.read_csv(INPUT)
    if "code" not in df.columns:
        raise KeyError("Input CSV must contain a 'code' column.")

    # Prepare text for both branches
    df["code_tfidf"] = df["code"].apply(clean_for_tfidf)
    df["code_feats"] = df["code"].apply(clean_for_feats)

    # TF-IDF transform
    X_tfidf = vect.transform(df["code_tfidf"])

    # Dense features: build -> align -> scale
    F = build_dense_features_df(df["code_feats"])
    F = align_dense_features(F, DENSE_FEATURE_NAMES)
    F = F.fillna(0.0)

    # Safety check on scaler dims
    expected = getattr(scaler, "mean_", None)
    if expected is not None and F.shape[1] != expected.shape[0]:
        raise ValueError(
            f"Dense feature count ({F.shape[1]}) ≠ scaler size ({expected.shape[0]}). "
            f"Ensure you used the same feature set/order as training."
        )
    F_scaled = scaler.transform(F.values.astype(np.float32))

    # Stack
    X = hstack([X_tfidf, csr_matrix(F_scaled)], format="csr")
    X, = ensure_csr(X)

    # Predict
    prob_human = sk_model.predict_proba(X)[:, 1] 
    pred_enc = sk_model.predict(X)
    pred_label = le.inverse_transform(pred_enc)

    # Output frame (include optional cols if present)
    cols = {}
    if "task_name" in df.columns: cols["task_name"] = df["task_name"]
    out = pd.DataFrame({
        **cols,
        "pred_label": pred_label,
        "prob_human": prob_human,
    })

    # Add true labels and correctness if available
    if "label" in df.columns:
        out["true_label"] = df["label"]
        out["correct"] = np.where(out["pred_label"] == out["true_label"], "CorrectPred", "IncorrectPred")

    # write output
    out.to_csv(OUTPUT, index=False)

    # print summary
    print(f"[OK] Wrote predictions to {OUTPUT}")
    print(f"Shapes — TF-IDF: {X_tfidf.shape}, Dense: {F.shape}, Stacked: {X.shape}")

if __name__ == '__main__':
    main()