# src/models/xgb_tfidf/train_xgb.py

import os
import joblib
import argparse
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an XGBoost classifier on TF-IDF features"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/processed/tfidf",
        help="Directory with train.pkl, validation.pkl, test.pkl"
    )
    parser.add_argument(
        "--model_out", type=str, default="models/xgb_tfidf/xgb_baseline.json",
        help="Path to save the trained XGBoost model"
    )
    parser.add_argument(
        "--n_estimators", type=int, default=300,
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.1,
    )
    parser.add_argument(
        "--max_depth", type=int, default=4,
    )
    parser.add_argument(
        "--subsample", type=float, default=0.8,
    )
    parser.add_argument(
        "--colsample_bytree", type=float, default=0.5,
    )
    parser.add_argument(
        "--early_stopping_rounds", type=int, default=50,
    )
    parser.add_argument(
        "--n_jobs", type=int, default=4
    )
    return parser.parse_args()

def ensure_csr(*mats):
    from scipy.sparse import csr_matrix, issparse, isspmatrix_csr
    out = []
    for M in mats:
        if not issparse(M):
            # If this happens, your TF-IDF pipeline produced dense arrays — fix upstream.
            # Converting dense->CSR won’t save RAM; consider dimensionality reduction.
            M = csr_matrix(M)
        elif not isspmatrix_csr(M):
            M = M.tocsr()
        out.append(M)
    return out

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    # Load data
    X_train, y_train = joblib.load(os.path.join(args.data_dir, "train.pkl"))
    X_val,   y_val   = joblib.load(os.path.join(args.data_dir, "validation.pkl"))
    X_test,  y_test  = joblib.load(os.path.join(args.data_dir, "test.pkl"))

    # Ensure sparse CSR matrices for memory efficiency
    X_train, X_val, X_test = ensure_csr(X_train, X_val, X_test)

    # Encode string labels to integers
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)

    # IMPORTANT: make sure train has both classes
    if len(set(y_train_enc)) < 2: # type: ignore
        raise ValueError("Train split contains <2 classes. Rebalance or adjust your split.")

    # Initialize classifier
    clf = XGBClassifier(
        objective='binary:logistic',
        # use_label_encoder=False,
        eval_metric='auc',
        tree_method='hist', # 'hist' is efficient for large datasets
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        early_stopping_rounds=args.early_stopping_rounds,
        reg_lambda=1.0,
        n_jobs=args.n_jobs,
        random_state=42,
    )

    # Fit with early stopping
    clf.fit(
        X_train, y_train_enc,
        eval_set=[(X_val, y_val_enc)],
        verbose=True,
    )

    # Save model and label encoder
    clf.get_booster().save_model(args.model_out)
    joblib.dump(le, os.path.splitext(args.model_out)[0] + '_label_encoder.pkl')
    print(f"Model saved to {args.model_out}")

    # Evaluate
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)    
    y_pred_labels = le.inverse_transform(y_pred)

    # Calculate ROC-AUC and classification report
    auc = roc_auc_score(y_test_enc, y_proba)
    print(f"Test ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred_labels))

if __name__ == '__main__':
    main()
