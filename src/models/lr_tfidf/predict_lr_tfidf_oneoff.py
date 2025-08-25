# src/models/lr_tfidf/predict_lr_tfidf_oneoff.py

import json
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from src.features.python_code_features import python_code_features

VECT_PATH   = "data/processed/tfidf/tfidf_vectorizer.pkl"
MODEL_PATH  = "models/lr_tfidf/lr_tfidf.pkl"
ENC_PATH    = "models/lr_tfidf/lr_tfidf_label_encoder.pkl"
SCALER_PATH = "models/lr_tfidf/lr_tfidf_scaler.pkl"
DENSE_NAMES = "models/lr_tfidf/lr_tfidf_dense_features.json"
THRESHOLD   = 0.5  # classification threshold

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
model = joblib.load(MODEL_PATH)

with open(DENSE_NAMES, "r", encoding="utf-8") as f:
    dense_cols = json.load(f)

pos_label = "Human_written"
pos_label_enc = le.transform([pos_label])[0] 
scale_dense = True

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

sk_classes = model.classes_
if pos_label_enc not in sk_classes:
    raise ValueError(f"Encoded pos_label {pos_label_enc} not in model.classes_: {sk_classes}")
pos_idx = int(np.where(sk_classes == pos_label_enc)[0][0])

p_human = float(model.predict_proba(X)[:, pos_idx][0])
y_bit = int(p_human >= THRESHOLD)

# map 0/1 to the actual encoded class id, then decode
other_enc = int(sk_classes[0] if sk_classes[0] != pos_label_enc else sk_classes[1])
pred_enc = pos_label_enc if y_bit == 1 else other_enc
label = le.inverse_transform([pred_enc])[0]

# 4) Output
print(f"Predicted label: {label}")
print(f"P(human) = {p_human:.3f} | P(AI) = {1.0 - p_human:.3f}")