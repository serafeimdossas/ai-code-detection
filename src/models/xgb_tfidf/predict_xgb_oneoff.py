# src/models/xgb_tfidf/predict_xgb_oneoff.py

import json
import joblib
import numpy as np
import xgboost as xgb
from scipy.sparse import hstack, csr_matrix
from src.features.python_code_features import python_code_features

VECT_PATH   = "data/processed/tfidf/tfidf_vectorizer.pkl"
MODEL_PATH  = "models/xgb_tfidf/xgb_tfidf.json"
ENC_PATH    = "models/xgb_tfidf/xgb_tfidf_label_encoder.pkl"
SCALER_PATH = "models/xgb_tfidf/xgb_tfidf_scaler.pkl"
DENSE_NAMES = "models/xgb_tfidf/xgb_tfidf_dense_features.json"

# Cleaning functions

# clean the code snippet for TF-IDF vectorization
def clean_for_tfidf(snippet: str) -> str:
    if not isinstance(snippet, str):
        return ""
    return " ".join(snippet.split())

# clean the code snippet for feature extraction
def clean_for_feats(snippet: str) -> str:
    if not isinstance(snippet, str):
        return ""
    s = snippet.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in s.split("\n"))

# One-off code snippet to classify
snippet = r"""
"""

# 1) Load artifacts
vect = joblib.load(VECT_PATH)
le   = joblib.load(ENC_PATH)
scaler = joblib.load(SCALER_PATH)

with open(DENSE_NAMES, "r", encoding="utf-8") as f:
    dense_cols = json.load(f)

bst = xgb.Booster()
bst.load_model(MODEL_PATH)

# 2) Build both branches
# TF-IDF branch
code_tfidf = clean_for_tfidf(snippet)
X_tfidf = vect.transform([code_tfidf])  # shape (1, V)

# Dense engineered features branch
code_feats = clean_for_feats(snippet)
# dict of features
F = python_code_features(code_feats)
# Align to training order
row = np.array([[F.get(c, 0.0) for c in dense_cols]], dtype=np.float32)  # shape (1, D)
# Scale
row_scaled = scaler.transform(row)

# 3) Stack and predict
X = hstack([X_tfidf, csr_matrix(row_scaled)], format="csr")
dmat = xgb.DMatrix(X)

# probability for class encoded as 1
p_human = float(bst.predict(dmat)[0])
# threshold can be tuned
y_pred  = int(p_human >= 0.5)
label   = le.inverse_transform([y_pred])[0]

print(f"Predicted label: {label}")
print(f"P(human) = {p_human:.3f} | P(AI) = {1.0 - p_human:.3f}")
