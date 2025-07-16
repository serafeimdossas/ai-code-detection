import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# 1) Load your artifacts
vect = joblib.load("data/processed/tfidf_vectorizer.pkl")
bst  = xgb.Booster()
bst.load_model("models/xgb_baseline.json")
le   = joblib.load("models/xgb_baseline_label_encoder.pkl")

# 2) Your one-off code snippet
snippet = '''
	
from decimal import *
 
D = Decimal
getcontext().prec = 100
a = n = D(1)
g, z, half = 1 / D(2).sqrt(), D(0.25), D(0.5)
for i in range(18):
x = [(a + g) * half, (a * g).sqrt()]
var = x[0] - a
z -= var * var * n
n += n
a, g = x
print(a * a / z)
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
