# src/models/xgb_tfidf/predict_xgb_oneoff.py

import joblib
import xgboost as xgb

# 1) Load your artifacts
vect = joblib.load("data/processed/tfidf/tfidf_vectorizer.pkl")
bst  = xgb.Booster()
bst.load_model("models/xgb_tfidf/xgb_baseline.json")
le   = joblib.load("models/xgb_tfidf/xgb_baseline_label_encoder.pkl")

# 2) Your one-off code snippet
snippet = '''
'''

# 3) Clean & vectorize
clean = " ".join(snippet.split())                # same clean_code logic
X = vect.transform([clean])                      # note the list for a batch of size 1

# 4) Predict
dmat   = xgb.DMatrix(X)
proba  = bst.predict(dmat)[0]                    # probability of "Human_written"
pred_i = int(proba >= 0.5)                       # threshold at 0.5
label  = le.inverse_transform([pred_i])[0]

print(f"Predicted label: {label}")
print(f"P(human) = {proba:.3f}, P(AI) = {1-proba:.3f}")
