# src/models/xgb_tfidf/predict.py

import os
import argparse
import json
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="Predict human vs. AI code snippets using trained XGBoost model")
    parser.add_argument(
        "--model", type=str, default="models/xgb_tfidf/xgb_code_features_baseline.json", 
        help="Path to the trained XGBoost model file"
    )
    parser.add_argument(
        "--vectorizer", type=str, default="data/processed/tfidf/tfidf_vectorizer.pkl", 
        help="Path to the fitted TF-IDF vectorizer pickle"
    )
    parser.add_argument(
        "--label_encoder", type=str, default="models/xgb_tfidf/xgb_code_features_baseline_label_encoder.pkl", 
        help="Path to the LabelEncoder pickle"
    )
    parser.add_argument(
        "--scaler", type=str, default="models/xgb_tfidf/xgb_code_features_baseline_scaler.pkl", 
        help="StandardScaler for dense features"
    )
    parser.add_argument(
        "--dense_feature_names", type=str, default="models/xgb_tfidf/xgb_code_features_baseline_dense_features.json", 
        help="JSON list with dense feature columns used at training (ordered)"
    )
    parser.add_argument(
        "--input", type=str, default="data/raw/test.csv", 
        help="Path to a CSV file with a column 'code' containing snippets to predict"
    )
    parser.add_argument(
        "--output", type=str, default="output/xgb_tfidf_code_features_predictions.csv", 
        help="Path to write predictions CSV"
    )
    return parser.parse_args()

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

# Import the feature extraction function
from src.features.python_code_features import python_code_features

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

# Load model from either XGBoost or scikit-learn format
def load_model_any(path: str):
    # Check if the path is a pickle file or an XGBoost model
    if path.endswith(".pkl"):
        return joblib.load(path), None
    import xgboost as xgb
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
    vect = joblib.load(args.vectorizer)
    le = joblib.load(args.label_encoder)
    scaler = joblib.load(args.scaler)
    sk_model, booster = load_model_any(args.model)

    # Input
    df = pd.read_csv(args.input)
    if "code" not in df.columns:
        raise KeyError("Input CSV must contain a 'code' column.")

    # Clean for each branch
    df["code_tfidf"] = df["code"].apply(clean_for_tfidf)
    df["code_feats"] = df["code"].apply(clean_for_feats)

    # TF-IDF transform
    X_tfidf = vect.transform(df["code_tfidf"])

    # Dense features: build -> align -> scale
    F = build_dense_features_df(df["code_feats"])
    F = align_dense_features(F, args.dense_feature_names)

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
    if sk_model is not None:
        prob_human = sk_model.predict_proba(X)[:, 1]
        pred_enc = sk_model.predict(X)
    else:
        import xgboost as xgb
        dmat = xgb.DMatrix(X)
        prob_human = booster.predict(dmat) # type: ignore
        pred_enc = (prob_human >= 0.5).astype(int)

    pred_label = le.inverse_transform(pred_enc)

    # Output frame (include optional cols if present)
    cols = {}
    if "task_name" in df.columns: cols["task_name"] = df["task_name"]
    out = pd.DataFrame({
        **cols,
        "pred_label": pred_label,
        "prob_human": prob_human,
    })

    if "label" in df.columns:
        out["true_label"] = df["label"]
        out["correct"] = np.where(out["pred_label"] == out["true_label"], "CorrectPred", "IncorrectPred")

    out.to_csv(args.output, index=False)
    print(f"[OK] Wrote predictions to {args.output}")
    print(f"Shapes — TF-IDF: {X_tfidf.shape}, Dense: {F.shape}, Stacked: {X.shape}")

if __name__ == '__main__':
    main()
