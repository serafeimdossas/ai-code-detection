# src/models/xgb_codebert/train_xgb_emb.py

import os
import joblib
import json
import numpy as np
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_DIR = "data/processed/codebert"
FEATURES_DIR = "data/processed/features"
MODEL_OUT = "models/xgb_codebert/xgb_codebert.json"
N_ESTIMATORS = 500
LEARNING_RATE = 0.1
MAX_DEPTH = 6
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
REG_ALPHA = 0.0
REG_LAMBDA = 1.0
EARLY_STOPPING_ROUNDS = 20
N_JOBS = 4
USE_GPU = False

# load embeddings and labels
def load_split_embeddings_labels(data_dir, split):
    X = np.load(os.path.join(data_dir, f"{split}_emb.npy"), allow_pickle=True)
    y = np.load(os.path.join(data_dir, f"{split}_labels.npy"), allow_pickle=True)
    return X, y

# load code features
def load_dense_features_df(features_dir, split):
    # expects pandas DataFrame pickled via joblib.dump
    F = joblib.load(os.path.join(features_dir, f"{split}_dense_features.pkl"))
    # ensure numeric float32
    return F.astype("float32")

def main():
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    # load embeddings and labels
    X_train_emb, y_train = load_split_embeddings_labels(DATA_DIR, "train")
    X_val_emb,   y_val   = load_split_embeddings_labels(DATA_DIR, "validation")
    X_test_emb,  y_test  = load_split_embeddings_labels(DATA_DIR, "test")

    # load engineered code features
    F_train = load_dense_features_df(FEATURES_DIR, "train")
    F_val   = load_dense_features_df(FEATURES_DIR, "validation")
    F_test  = load_dense_features_df(FEATURES_DIR, "test")

    # Save dense feature names to a JSON file
    dense_names_out = os.path.splitext(MODEL_OUT)[0] + "_dense_feature_names.json"
    with open(dense_names_out, "w", encoding="utf-8") as f:
        json.dump(list(F_train.columns), f, indent=2)

    # scale code features
    scaler = StandardScaler(with_mean=True, with_std=True)
    F_train_scaled = scaler.fit_transform(F_train.values)
    F_val_scaled   = scaler.transform(F_val.values)
    F_test_scaled  = scaler.transform(F_test.values)

    # Save the scaler for inference/serving
    scaler_out = os.path.splitext(MODEL_OUT)[0] + "_scaler.pkl"
    joblib.dump(scaler, scaler_out)
    print(f"[INFO] Saved scaler -> {scaler_out}")

    # stack embeddings with code features
    X_train = np.hstack([X_train_emb, F_train_scaled]).astype("float32")
    X_val   = np.hstack([X_val_emb,   F_val_scaled]).astype("float32")
    X_test  = np.hstack([X_test_emb,  F_test_scaled]).astype("float32")

    # encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)

    # Save class mapping for reference
    classes_ = list(le.classes_)

    # handle class imbalance
    pos = max(1, int((y_train_enc == 1).sum())) # type: ignore
    neg = max(1, int((y_train_enc == 0).sum())) # type: ignore
    scale_pos_weight = float(neg) / float(pos)

    # Initialize XGBoost classifier
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="gpu_hist" if USE_GPU else "hist",
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        reg_alpha=REG_ALPHA,
        reg_lambda=REG_LAMBDA,
        n_jobs=N_JOBS,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        max_bin=256,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    # Train the model
    clf.fit(
        X_train, y_train_enc,
        eval_set=[(X_val, y_val_enc)],
        verbose=True,
    )

    # Save model + encoder
    clf.get_booster().save_model(MODEL_OUT)
    joblib.dump(le, os.path.splitext(MODEL_OUT)[0] + "_label_encoder.pkl")
    print(f"[OK] Model saved -> {MODEL_OUT}")

    # evaluate
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred_enc = (y_proba >= 0.5).astype(int)
    y_pred_labels = le.inverse_transform(y_pred_enc)

    # Compute metrics: ROC-AUC, classification report, confusion matrix
    auc = roc_auc_score(y_test_enc, y_proba)
    report = classification_report(y_test, y_pred_labels, output_dict=True, digits=4)
    cm = confusion_matrix(y_test_enc, y_pred_enc).tolist()

    print(f"Test ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred_labels, digits=4))

    # Save evaluation artifacts with timestamp
    stamp = datetime.now().strftime("%d-%m-%Y")
    base = os.path.splitext(MODEL_OUT)[0]
    with open(base + f"_metrics_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump({"auc": auc, "report": report, "classes": classes_, "cm": cm}, f, indent=2)

if __name__ == '__main__':
    main()
