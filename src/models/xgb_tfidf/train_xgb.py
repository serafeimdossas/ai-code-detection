# src/models/xgb_tfidf/train_xgb.py

import os
import joblib
import json
import numpy as np
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.sparse import hstack, csr_matrix, issparse, isspmatrix_csr

DATA_DIR = "data/processed/tfidf"
FEATURES_DIR = "data/processed/features"
MODEL_OUT = "models/xgb_tfidf/xgb_tfidf.json"
N_ESTIMATORS = 300
LEARNING_RATE = 0.1
MAX_DEPTH = 4
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.5
REG_LAMBDA = 1.0
EARLY_STOPPING_ROUNDS = 50
N_JOBS = 4
USE_GPU = False

def ensure_csr(*mats):
    """
    Ensure that all input matrices are in CSR (Compressed Sparse Row) format.
    - If a matrix is dense (NumPy array), it will be converted to CSR.
    - If a matrix is sparse but not CSR it will be converted to CSR.
    - If it's already CSR, it is left unchanged.
    """
    out = []
    for M in mats:
        if not issparse(M):
            # Matrix is dense (NumPy array, not sparse).
            M = csr_matrix(M)
        elif not isspmatrix_csr(M):
            # Matrix is already sparse but not in CSR format.
            M = M.tocsr()
        # Collect the ensured-CSR matrix
        out.append(M)
    return out

def add_dense_feats(X_sparse, feats_path, scaler=None, fit=False):
    """
    Load dense engineered features (DataFrame), scale them, and hstack with X_sparse.
    Returns (X_combined, scaler).
    """
    
    # Load dense engineered features
    F = joblib.load(feats_path)  # expected: pandas DataFrame
    if not hasattr(F, "values"):
        # the loaded object must be a pandas DataFrame
        raise ValueError(f"{feats_path} does not look like a pandas DataFrame. Got: {type(F)}")
    
    F_mat = F.values.astype(np.float32)

    # Scale the dense features
    if fit:
        # Fit a new StandardScaler and apply it to training features
        scaler = StandardScaler(with_mean=True, with_std=True)
        F_scaled = scaler.fit_transform(F_mat)
    else:
        # For validation/test, reuse the existing scaler
        if scaler is None:
            raise ValueError("Scaler must be provided when fit=False.")
        F_scaled = scaler.transform(F_mat)

    # Stack sparse (X_sparse) with dense features (converted to sparse CSR matrix)
    X_combined = hstack([X_sparse, csr_matrix(F_scaled)], format="csr")
    return X_combined, scaler

def main():
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    # Load TF-IDF (sparse) and labels
    X_train_tfidf, y_train = joblib.load(os.path.join(DATA_DIR, "train.pkl"))
    X_val_tfidf,   y_val   = joblib.load(os.path.join(DATA_DIR, "validation.pkl"))
    X_test_tfidf,  y_test  = joblib.load(os.path.join(DATA_DIR, "test.pkl"))

    # Load engineered dense features and combine with TF-IDF (scale on train, reuse for val/test)
    train_feats_path = os.path.join(FEATURES_DIR, "train_dense_features.pkl")
    val_feats_path   = os.path.join(FEATURES_DIR, "validation_dense_features.pkl")
    test_feats_path  = os.path.join(FEATURES_DIR, "test_dense_features.pkl")

    # Load dense feature names for saving later
    dense_feature_names = joblib.load(train_feats_path).columns.tolist()

    # Save dense feature names to a JSON file
    feat_names_out = os.path.splitext(MODEL_OUT)[0] + "_dense_features.json"
    with open(feat_names_out, "w") as f:
        json.dump(dense_feature_names, f)

    print(f"Dense feature names saved to {feat_names_out}")

    X_train, scaler = add_dense_feats(X_train_tfidf, train_feats_path, fit=True)
    X_val,   _      = add_dense_feats(X_val_tfidf,   val_feats_path,   scaler=scaler, fit=False)
    X_test,  _      = add_dense_feats(X_test_tfidf,  test_feats_path,  scaler=scaler, fit=False)

    # Save the scaler for inference/serving
    scaler_out = os.path.splitext(MODEL_OUT)[0] + "_scaler.pkl"
    joblib.dump(scaler, scaler_out)

    # Ensure sparse CSR matrices for memory efficiency
    X_train, X_val, X_test = ensure_csr(X_train, X_val, X_test)

    # Encode string labels to integers
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)

    # Save class mapping for reference
    classes_ = list(le.classes_)
    print(f"Label mapping: {dict(zip(classes_, le.transform(classes_)))}") # type: ignore

    # make sure train has both classes
    if len(set(y_train_enc)) < 2: # type: ignore
        raise ValueError("Train split contains <2 classes. Rebalance or adjust your split.")

    # Class imbalance handling (optional but helpful)
    pos = int((y_train_enc == 1).sum()) # type: ignore
    neg = int((y_train_enc == 0).sum()) # type: ignore
    scale_pos_weight = float(neg) / float(max(pos, 1))
    print(f"scale_pos_weight (class 1): {scale_pos_weight:.3f}")

    # Initialize XGBoost classifier
    clf = XGBClassifier(
        objective='binary:logistic',
        # use_label_encoder=False,
        eval_metric='auc',
        tree_method='gpu_hist' if USE_GPU else 'hist', # 'hist' is efficient for large datasets
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        reg_lambda=1.0,
        n_jobs=N_JOBS,
        random_state=42,
    )

    # Train the model
    clf.fit(
        X_train, y_train_enc,
        eval_set=[(X_val, y_val_enc)],
        verbose=True,
    )

    # Save model and label encoder
    clf.get_booster().save_model(MODEL_OUT)
    joblib.dump(le, os.path.splitext(MODEL_OUT)[0] + '_label_encoder.pkl')
    print(f"Model saved to {MODEL_OUT}")

    # Evaluate
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred_enc = (y_proba >= 0.5).astype(int) 
    y_pred = le.inverse_transform(y_pred_enc)

    # Compute metrics: ROC-AUC, classification report, confusion matrix
    auc = roc_auc_score(y_test_enc, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True, digits=4)
    cm = confusion_matrix(y_test_enc, y_pred_enc).tolist()

    print(f"Test ROC-AUC: {auc:.4f}")
    print(json.dumps(report, indent=2))

    # Save evaluation artifacts with timestamp
    stamp = datetime.now().strftime("%d-%m-%Y")
    base = os.path.splitext(MODEL_OUT)[0]
    with open(base + f"_metrics_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump({"auc": auc, "report": report, "classes": classes_, "cm": cm}, f, indent=2)

if __name__ == '__main__':
    main()
