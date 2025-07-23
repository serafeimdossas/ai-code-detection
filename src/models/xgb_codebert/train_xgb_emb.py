# src/models/xgb_codebert/train_xgb_emb.py

import os
import joblib
import argparse
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train XGBoost classifier on precomputed embedding features"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/processed/codebert",
        help="Directory containing train_emb.npy, train_labels.npy, validation_emb.npy, validation_labels.npy, test_emb.npy, test_labels.npy"
    )
    parser.add_argument(
        "--model_out", type=str, default="models/xgb_codebert/xgb_with_emb.json",
        help="Output path for the trained XGBoost model"
    )
    parser.add_argument(
        "--n_estimators", type=int, default=500,
        help="Number of boosting rounds"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.1,
        help="Learning rate (eta)"
    )
    parser.add_argument(
        "--max_depth", type=int, default=6,
        help="Maximum tree depth"
    )
    parser.add_argument(
        "--subsample", type=float, default=0.8,
        help="Subsample ratio of the training instances"
    )
    parser.add_argument(
        "--colsample_bytree", type=float, default=0.8,
        help="Subsample ratio of columns when constructing each tree"
    )
    parser.add_argument(
        "--early_stopping_rounds", type=int, default=20,
        help="Rounds of early stopping"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    # Load embeddings and labels
    X_train = np.load(os.path.join(args.data_dir, 'train_emb.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(args.data_dir, 'train_labels.npy'), allow_pickle=True)
    X_val   = np.load(os.path.join(args.data_dir, 'validation_emb.npy'), allow_pickle=True)
    y_val   = np.load(os.path.join(args.data_dir, 'validation_labels.npy'), allow_pickle=True)
    X_test  = np.load(os.path.join(args.data_dir, 'test_emb.npy'), allow_pickle=True)
    y_test  = np.load(os.path.join(args.data_dir, 'test_labels.npy'), allow_pickle=True)

    # Encode labels if they are strings
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)

    # Initialize classifier
    clf = XGBClassifier(
        objective='binary:logistic',
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        eval_metric='auc',
        use_label_encoder=False,
        early_stopping_rounds=args.early_stopping_rounds
    )

    # Train with early stopping on validation set
    clf.fit(
        X_train, y_train_enc,
        eval_set=[(X_val, y_val_enc)],
        verbose=True
    )

    # Save model and encoder
    clf.get_booster().save_model(args.model_out)
    joblib.dump(le, os.path.splitext(args.model_out)[0] + '_label_encoder.pkl')
    print(f"Model saved to {args.model_out}")

    # Evaluate on test set
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    y_pred_labels = le.inverse_transform(y_pred)

    auc = roc_auc_score(y_test_enc, y_proba)
    print(f"Test ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred_labels))


if __name__ == '__main__':
    main()
