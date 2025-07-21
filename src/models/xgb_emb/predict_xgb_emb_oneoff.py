# src/models/xgb_emb/predict_xgb_emb_oneoff.py

import joblib
import xgboost as xgb

# 1) Load your artifacts
vect = joblib.load("data/processed/tfidf/tfidf_vectorizer.pkl")
bst  = xgb.Booster()
bst.load_model("models/xgb_emb/xgb_with_emb.json")
le   = joblib.load("models/xgb_emb/xgb_with_emb_label_encoder.pkl")

# 2) Your one-off code snippet
snippet = '''
def fibonacci(n):
    """
    Return the n-th Fibonacci number, where:
      fibonacci(0) == 0
      fibonacci(1) == 1

    Raises:
        ValueError: if n is negative.
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
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
