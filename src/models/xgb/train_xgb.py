# src/models/xgb/train_xgb.py

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
        "--data_dir", type=str, default="data/processed",
        help="Directory with train.pkl, validation.pkl, test.pkl"
    )
    parser.add_argument(
        "--model_out", type=str, default="models/xgb_baseline.json",
        help="Path to save the trained XGBoost model"
    )
    parser.add_argument(
        "--n_estimators", type=int, default=500,
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.1,
    )
    parser.add_argument(
        "--max_depth", type=int, default=6,
    )
    parser.add_argument(
        "--subsample", type=float, default=0.8,
    )
    parser.add_argument(
        "--colsample_bytree", type=float, default=0.8,
    )
    parser.add_argument(
        "--early_stopping_rounds", type=int, default=20,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    # Load data
    X_train, y_train = joblib.load(os.path.join(args.data_dir, "train.pkl"))
    X_val,   y_val   = joblib.load(os.path.join(args.data_dir, "validation.pkl"))
    X_test,  y_test  = joblib.load(os.path.join(args.data_dir, "test.pkl"))

    # Encode string labels to integers
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)

    # Initialize classifier
    clf = XGBClassifier(
        objective='binary:logistic',
        # use_label_encoder=False,
        eval_metric='auc',
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    # Fit with early stopping
    clf.fit(
        X_train, y_train_enc,
        eval_set=[(X_val, y_val_enc)],
        verbose=True
    )

    # Save model and label encoder
    clf.get_booster().save_model(args.model_out)
    joblib.dump(le, os.path.splitext(args.model_out)[0] + '_label_encoder.pkl')
    print(f"Model saved to {args.model_out}")

    # Evaluate on test set
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    # Inverse transform predictions to original labels
    y_pred_labels = le.inverse_transform(y_pred)

    auc = roc_auc_score(y_test_enc, y_proba)
    print(f"Test ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred_labels))

if __name__ == '__main__':
    main()
