# src/models/xgb_emb/predict_xgb_emb_oneoff.py

import joblib
import xgboost as xgb
import numpy as np
from sentence_transformers import SentenceTransformer

# 1) Load your artifacts
# Load embedding model
embed_model = SentenceTransformer("microsoft/codebert-base")

# Load trained XGBoost booster
bst = xgb.Booster()
bst.load_model("models/xgb_emb/xgb_with_emb.json")

# Load label encoder
le = joblib.load("models/xgb_emb/xgb_with_emb_label_encoder.pkl")

# 2) Your one-off code snippet
tn = '''
'''

# 3) Clean & embed
def clean_code(snippet: str) -> str:
    return " ".join(snippet.split())

clean = clean_code(tn)
emb = embed_model.encode([clean], batch_size=1)

# Convert to DMatrix
dmat = xgb.DMatrix(np.array(emb))

# 4) Predict
# Probability of class 'Human_written' (encoded as 1)
proba = bst.predict(dmat)[0]
pred_i = int(proba >= 0.5)
label = le.inverse_transform([pred_i])[0]

print(f"Predicted label: {label}")
print(f"P(human) = {proba:.3f}, P(AI) = {1 - proba:.3f}")
