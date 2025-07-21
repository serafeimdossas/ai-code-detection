# src/models/mlp/predict_oneoff_mlp.py

import joblib
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from train_mlp_emb import MLP

# 1) Load your artifacts
# Path to your MLP state dict
state_dict = torch.load("models/mlp_emb/mlp_emb.pt")

# Load label encoder
le = joblib.load("models/mlp_emb/mlp_emb_label_encoder.pkl")

# 2) Initialize embedding model and MLP
embed_model = SentenceTransformer("microsoft/codebert-base")

# Determine embedding dimension using a dummy input
dim = np.array(embed_model.encode(["dummy"], batch_size=1)).shape[1]
model = MLP(input_dim=dim)
model.load_state_dict(state_dict)
model.eval()

# 3) Your one-off code snippet
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

# 4) Clean & embed
clean = " ".join(snippet.split())
emb = embed_model.encode([clean], batch_size=1)
X = torch.from_numpy(np.array(emb)).float()

# 5) Predict
with torch.no_grad():
    proba = model(X).squeeze().item()  # P(human_written)
pred_idx = int(proba >= 0.5)
label = le.inverse_transform([pred_idx])[0]

print(f"Predicted label: {label}")
print(f"P(human) = {proba:.3f}, P(AI) = {1 - proba:.3f}")
