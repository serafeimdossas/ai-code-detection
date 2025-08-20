# src/models/xgb_codebert/predict_xgb_emb_oneoff.py

import json
import joblib
import xgboost as xgb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from src.features.python_code_features import python_code_features

MODEL_PATH   = "models/xgb_codebert/xgb_codebert.json"
ENC_PATH     = "models/xgb_codebert/xgb_codebert_label_encoder.pkl"
SCALER_PATH  = "models/xgb_codebert/xgb_codebert_scaler.pkl"
NAMES_PATH   = "models/xgb_codebert/xgb_codebert_dense_feature_names.json"
EMBED_MODEL  = "microsoft/codebert-base"

THRESHOLD = 0.50

# Cleaning functions

# clean the code snippet for embedding
def clean_for_embed(snippet: str) -> str:
    if not isinstance(snippet, str):
        return ""
    return snippet.replace("\r\n", "\n").replace("\r", "\n")

# clean the code snippet for feature extraction
def clean_for_feats(snippet: str) -> str:
    if not isinstance(snippet, str):
        return ""
    s = snippet.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in s.split("\n"))

# One-off code snippet to classify
snippet = r"""
"""

# 1) Load your artifacts, the model, label encoder, and scaler
le = joblib.load(ENC_PATH)
scaler: StandardScaler = joblib.load(SCALER_PATH)
with open(NAMES_PATH, "r", encoding="utf-8") as f:
    dense_cols = json.load(f)

sk_model = None
if MODEL_PATH.endswith(".pkl"):
    sk_model = joblib.load(MODEL_PATH)
    booster = None
else:
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)

embedder = SentenceTransformer(EMBED_MODEL)

# 2) Build embedding and features
# Code embedding branch
code_embed = clean_for_embed(snippet)
emb = embedder.encode(
    [code_embed],
    batch_size=1,
    convert_to_numpy=True,
    normalize_embeddings=False
).astype("float32") 

# Dense engineered features branch
code_feats = clean_for_feats(snippet)
F_dict = python_code_features(code_feats)
row_dense = np.array([[F_dict.get(c, 0.0) for c in dense_cols]], dtype=np.float32)
row_dense_scaled = scaler.transform(row_dense) 

# Stack
X = np.hstack([emb, row_dense_scaled]).astype("float32")

# 3) Predict
if sk_model is not None:
    p_pos = float(sk_model.predict_proba(X)[:, 1][0])
else:
    dmat = xgb.DMatrix(X)
    p_pos = float(booster.predict(dmat)[0]) # type: ignore

pred_i = int(p_pos >= THRESHOLD)
label = le.inverse_transform([pred_i])[0]

print(f"Predicted label: {label}")
print(f"P(human) = {p_pos:.3f} | P(AI) = {1.0 - p_pos:.3f}")
