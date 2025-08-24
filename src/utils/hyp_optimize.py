import os
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix, issparse, isspmatrix_csr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

TFIDF_DIR = "data/processed/tfidf"
CODE_FEATURES_DIR = "data/processed/features"
THRESHOLD = 0.5
MODEL = "models/lr_tfidf_test/lr_tfidf_test.pkl"

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
    # Load TF-IDF (sparse) and labels
    X_train_tfidf, y_train = joblib.load(os.path.join(TFIDF_DIR, "train.pkl"))
    X_val_tfidf,   y_val   = joblib.load(os.path.join(TFIDF_DIR, "validation.pkl"))
    X_test_tfidf,  y_test  = joblib.load(os.path.join(TFIDF_DIR, "test.pkl"))

    # Load engineered dense features and combine with TF-IDF (scale on train, reuse for val/test)
    train_feats_path = os.path.join(CODE_FEATURES_DIR, "train_dense_features.pkl")
    val_feats_path   = os.path.join(CODE_FEATURES_DIR, "validation_dense_features.pkl")
    test_feats_path  = os.path.join(CODE_FEATURES_DIR, "test_dense_features.pkl")

    X_train, scaler = add_dense_feats(X_train_tfidf, train_feats_path, fit=True)
    X_val,   _      = add_dense_feats(X_val_tfidf,   val_feats_path,   scaler=scaler, fit=False)
    X_test,  _      = add_dense_feats(X_test_tfidf,  test_feats_path,  scaler=scaler, fit=False)

    # Ensure sparse CSR matrices for memory efficiency
    X_train, X_val, X_test = ensure_csr(X_train, X_val, X_test)

    # Label encode
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)

    # --- 1) Define base estimator (sparse-friendly) ---
    base_lr = LogisticRegression(
        penalty="l2",         # start simple; you can try elasticnet later
        max_iter=5000,
        n_jobs=-1
    )

    # --- 2) Param distributions (log-uniform over C; tune tol and class_weight) ---
    param_dist = {
        "C": np.logspace(-1, 1, 20),   # 0.1 to 10
        "tol": [1e-2, 3e-3, 1e-3, 3e-4, 1e-4],
        "class_weight": [None, "balanced"],
        "solver": ["liblinear", "saga"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=base_lr,
        param_distributions=param_dist,
        n_iter=30,                   # bump to 80+ if you have time
        scoring="roc_auc",           # ROC-AUC is robust for balanced classes
        cv=cv,
        n_jobs=-1,
        refit=True,                  # refit on full train with best params
        verbose=1,
        random_state=42
    )

    # --- 3) Run search on TRAIN only ---
    search.fit(X_train, y_train_enc)
    best_lr = search.best_estimator_
    print("Best params:", search.best_params_)
    print("CV ROC-AUC:", search.best_score_)

if __name__ == '__main__':
    main()