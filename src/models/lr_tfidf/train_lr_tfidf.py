import os
import joblib
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.sparse import hstack, csr_matrix, issparse, isspmatrix_csr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

MODEL="models/lr_tfidf/lr_tfidf.pkl"
LABEL_ENCODER="models/lr_tfidf/lr_tfidf_label_encoder.pkl"
PENALTY="l2"  # or "none"
C=10.0
SOLVER="liblinear"  # "liblinear", "lbfgs", "saga", "newton-cg", "sag"
MAX_ITER=1000
CLASS_WEIGHT="balanced"  # or None
N_JOBS=-1  # for solvers that support parallelism, -1 means "use all processors"
TFIDF_DIR = "data/processed/tfidf"
CODE_FEATURES_DIR = "data/processed/features"
THRESHOLD = 0.5

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

def main():
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)

    # Load TF-IDF (sparse) and labels
    X_train_tfidf, y_train = joblib.load(os.path.join(TFIDF_DIR, "train.pkl"))
    X_val_tfidf,   y_val   = joblib.load(os.path.join(TFIDF_DIR, "validation.pkl"))
    X_test_tfidf,  y_test  = joblib.load(os.path.join(TFIDF_DIR, "test.pkl"))

    # Load engineered dense features and combine with TF-IDF (scale on train, reuse for val/test)
    train_feats_path = os.path.join(CODE_FEATURES_DIR, "train_dense_features.pkl")
    val_feats_path   = os.path.join(CODE_FEATURES_DIR, "validation_dense_features.pkl")
    test_feats_path  = os.path.join(CODE_FEATURES_DIR, "test_dense_features.pkl")

    # Load dense feature names for saving later
    dense_feature_names = joblib.load(train_feats_path).columns.tolist()

    # Save dense feature names to a JSON file
    feat_names_out = os.path.splitext(MODEL)[0] + "_dense_features.json"
    with open(feat_names_out, "w") as f:
        json.dump(dense_feature_names, f)

    print(f"Dense feature names saved to {feat_names_out}")

    X_train, scaler = add_dense_feats(X_train_tfidf, train_feats_path, fit=True)
    X_val,   _      = add_dense_feats(X_val_tfidf,   val_feats_path,   scaler=scaler, fit=False)
    X_test,  _      = add_dense_feats(X_test_tfidf,  test_feats_path,  scaler=scaler, fit=False)

    # Save the scaler for inference/serving
    scaler_out = os.path.splitext(MODEL)[0] + "_scaler.pkl"
    joblib.dump(scaler, scaler_out)

    # Ensure sparse CSR matrices for memory efficiency
    X_train, X_val, X_test = ensure_csr(X_train, X_val, X_test)

    # Label encode
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)

    # Save class mapping for reference
    classes_ = list(le.classes_)
    print(f"Label mapping: {dict(zip(classes_, le.transform(classes_)))}") # type: ignore

    # initialize and train the Logistic Regression model
    lr = LogisticRegression(
        penalty=PENALTY,
        C=C,
        solver=SOLVER,
        max_iter=MAX_ITER,
        class_weight=CLASS_WEIGHT,
        n_jobs=N_JOBS,
        verbose=1,
        tol=3e-3,
    )

    # train the model
    lr.fit(X_train, y_train_enc)
    
    # Make sure output dir exists and save artifacts
    Path(os.path.dirname(MODEL)).mkdir(parents=True, exist_ok=True)
    joblib.dump(lr, MODEL)
    joblib.dump(le, LABEL_ENCODER)
    print(f"\nSaved LR model -> {MODEL}")
    print(f"Saved LabelEncoder -> {LABEL_ENCODER}")

    # Predict probabilities and classes
    y_proba = lr.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= THRESHOLD).astype(int)

    # Calculate metrics
    auc = roc_auc_score(y_test_enc, y_proba)
    report  = classification_report(y_test_enc, y_pred, output_dict=True, digits=4)
    cm  = confusion_matrix(y_test_enc, y_pred).tolist()
        
    # Print results
    print(f"Test ROC-AUC: {auc:.4f}")

    # Save evaluation artifacts with timestamp
    stamp = datetime.now().strftime("%d-%m-%Y")
    base = os.path.splitext(MODEL)[0]
    with open(base + f"_metrics_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump({"auc": auc, "report": report, "classes": classes_, "cm": cm}, f, indent=2)

if __name__ == '__main__':
    main()