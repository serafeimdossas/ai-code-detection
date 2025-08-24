# src/models/mlp_codebert/train_mlp_emb.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import numpy as np

EMB_DIR = "data/processed/codebert"
OUTPUT = "models/mlp_codebert/mlp_emb.pt"
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 10

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

def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    # Load embeddings & labels
    X_train = torch.from_numpy(np.load(f"{EMB_DIR}/train_emb.npy", allow_pickle=True)).float()
    y_train = np.load(f"{EMB_DIR}/train_labels.npy", allow_pickle=True)
    X_val   = torch.from_numpy(np.load(f"{EMB_DIR}/validation_emb.npy", allow_pickle=True)).float()
    y_val   = np.load(f"{EMB_DIR}/validation_labels.npy", allow_pickle=True)
    X_test  = torch.from_numpy(np.load(f"{EMB_DIR}/test_emb.npy", allow_pickle=True)).float()
    y_test  = np.load(f"{EMB_DIR}/test_labels.npy", allow_pickle=True)

    # Encode labels
    le = LabelEncoder().fit(y_train)
    y_train = torch.from_numpy(le.transform(y_train)).float()
    y_val   = torch.from_numpy(le.transform(y_val)).float()
    y_test  = torch.from_numpy(le.transform(y_test)).float()

    # DataLoaders
    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)
    test_ds  = TensorDataset(X_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # Model, optimizer, loss
    model = MLP(input_dim=X_train.size(1))
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()

    best_auc = 0.0
    best_path = OUTPUT
    # Training loop
    for epoch in range(EPOCHS):
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
            torch.save(model.state_dict(), OUTPUT)
            # Save label encoder separately
            joblib.dump(le, OUTPUT.replace('.pt', '_label_encoder.pkl'))

    print("Best val AUC:", best_auc)

    # Load best model weights
    model.load_state_dict(torch.load(OUTPUT))
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
