# src/models/mlp_codebert/train_mlp_emb.py

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_dir", default="data/processed/codebert")
    p.add_argument("--out", default="models/mlp_codebert/mlp_emb.pt")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Load embeddings & labels
    X_train = torch.from_numpy(np.load(f"{args.emb_dir}/train_emb.npy", allow_pickle=True)).float()
    y_train = np.load(f"{args.emb_dir}/train_labels.npy", allow_pickle=True)
    X_val   = torch.from_numpy(np.load(f"{args.emb_dir}/validation_emb.npy", allow_pickle=True)).float()
    y_val   = np.load(f"{args.emb_dir}/validation_labels.npy", allow_pickle=True)
    X_test  = torch.from_numpy(np.load(f"{args.emb_dir}/test_emb.npy", allow_pickle=True)).float()
    y_test  = np.load(f"{args.emb_dir}/test_labels.npy", allow_pickle=True)

    # Encode labels
    le = LabelEncoder().fit(y_train)
    y_train = torch.from_numpy(le.transform(y_train)).float()
    y_val   = torch.from_numpy(le.transform(y_val)).float()
    y_test  = torch.from_numpy(le.transform(y_test)).float()

    # DataLoaders
    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)
    test_ds  = TensorDataset(X_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size)

    # Model, optimizer, loss
    model = MLP(input_dim=X_train.size(1))
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

    best_auc = 0.0
    best_path = args.out
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            preds = model(xb).squeeze()
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = torch.cat([model(xb).squeeze() for xb, _ in val_dl]).cpu().numpy()
        auc = roc_auc_score(y_val.numpy(), val_preds)
        print(f"Epoch {epoch} val AUC {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            # Save only model weights
            torch.save(model.state_dict(), args.out)
            # Save label encoder separately
            joblib.dump(le, args.out.replace('.pt', '_label_encoder.pkl'))

    print("Best val AUC:", best_auc)

    # Load best model weights
    model.load_state_dict(torch.load(args.out))
    model.eval()
    with torch.no_grad():
        test_preds = torch.cat([model(xb).squeeze() for xb, _ in test_dl]).cpu().numpy()
    test_auc = roc_auc_score(y_test.numpy(), test_preds)
    print("Test ROC-AUC:", test_auc)
    # Classification report requires labels
    y_pred_labels = le.inverse_transform((test_preds >= 0.5).astype(int))
    y_true_labels = le.inverse_transform(y_test.numpy().astype(int))
    print(classification_report(y_true_labels, y_pred_labels))

if __name__=="__main__":
    main()
