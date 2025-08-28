"""
Time-Series Classification in PyTorch
------------------------------------

This single-file script trains a neural network (LSTM, 1D-CNN, or Transformer)
for labeled time-series classification. It includes:
  • Synthetic data generator (so you can run it immediately)
  • Dataset & DataLoader
  • Three model choices: LSTM, CNN, Transformer
  • Train/val split, early stopping, checkpointing
  • Metrics: accuracy & macro F1

USAGE (run from terminal):
    python timeseries_classifier.py --model lstm
    python timeseries_classifier.py --model cnn
    python timeseries_classifier.py --model transformer

To use your own data, replace `make_synthetic_dataset` with a loader that returns:
    X: FloatTensor of shape [N, T, D]
    y: LongTensor  of shape [N]
where N=#samples, T=#timesteps, D=#features.

Author: ChatGPT
"""
from __future__ import annotations
import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score

import pandas as pd
import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, csv_path, seq_len=30):
        df = pd.read_csv(csv_path)

        # Extract features & labels (drop time column)
        features = df[["feat1", "feat2", "feat3", "feat4", "feat5", "feat6"]].values
        labels = df["label"].values

        self.X, self.y = [], []
        for i in range(len(df) - seq_len + 1):
            window = features[i:i+seq_len]
            label = labels[i+seq_len-1]   # label = last timestep’s label
            self.X.append(window)
            self.y.append(label)

        self.X = torch.tensor(self.X, dtype=torch.float32)   # [N, T, D]
        self.y = torch.tensor(self.y, dtype=torch.long)      # [N]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Example:
# ds = CSVDataset("your_data.csv", seq_len=30)
# print(ds[0][0].shape)   # torch.Size([30, 6]) -> 30 timesteps, 6 features
# print(ds[0][1])         # label (0 or 1)


# ==========================
# Utilities
# ==========================

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Config:
    model: str = "lstm"            # one of {lstm, cnn, transformer}
    input_dim: int = 3             # features per timestep
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True     # lstm only

    num_heads: int = 4             # transformer only
    ff_dim: int = 256              # transformer only

    num_classes: int = 3
    seq_len: int = 150

    batch_size: int = 64
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4

    early_stop_patience: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "checkpoint.pt"


# ==========================
# Synthetic dataset (replace with your own loader)
# ==========================
class TimeSeriesDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert X.ndim == 3, "X must be [N, T, D]"
        assert y.ndim == 1, "y must be [N]"
        assert X.shape[0] == y.shape[0], "N mismatch"
        self.X = X.float()
        self.y = y.long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_synthetic_dataset(N: int, T: int, D: int, C: int, seed: int = 1337) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a simple multi-class dataset with distinct dynamics per class.
    Class 0: noisy sin waves
    Class 1: noisy sawtooth
    Class 2: AR(1) process
    Additional classes will recycle styles.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((N, T, D), dtype=np.float32)
    y = np.zeros(N, dtype=np.int64)

    def sin_wave(T, f=0.05):
        t = np.arange(T)
        return np.sin(2 * np.pi * f * t)

    def sawtooth(T, p=30):
        t = np.arange(T)
        return 2 * (t % p) / p - 1

    def ar1(T, phi=0.8):
        x = np.zeros(T)
        for i in range(1, T):
            x[i] = phi * x[i - 1] + rng.normal(0, 0.3)
        return x

    base_patterns = [sin_wave, sawtooth, ar1]

    for i in range(N):
        cls = i % C
        y[i] = cls
        pattern_fn = base_patterns[cls % len(base_patterns)]
        base = pattern_fn(T)
        features = []
        for d in range(D):
            noise = rng.normal(0, 0.15, size=T)
            scale = rng.uniform(0.8, 1.2)
            shift = rng.uniform(-0.5, 0.5)
            features.append(scale * base + noise + shift)
        X[i] = np.stack(features, axis=-1)

    # Standardize per-feature across the dataset
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-6
    X = (X - mean) / std
    return torch.from_numpy(X), torch.from_numpy(y)


# ==========================
# Models
# ==========================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.0, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes)
        )

    def forward(self, x):  # x: [B, T, D]
        out, (hn, cn) = self.lstm(x)
        # Use last hidden state from the top layer
        if self.lstm.bidirectional:
            last = torch.cat([hn[-2], hn[-1]], dim=-1)  # [B, 2H]
        else:
            last = hn[-1]
        logits = self.head(self.dropout(last))
        return logits


class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=128, dropout=0.1):
        super().__init__()
        # Expect input as [B, T, D]; we transpose to [B, D, T] for conv1d
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, D, T]
        h = self.conv(x).squeeze(-1)  # [B, hidden]
        h = self.dropout(h)
        return self.fc(h)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerEncoderClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, batch_first=True,
                                                   dropout=dropout, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        h = self.encoder(x)            # [B, T, d_model]
        h = h.mean(dim=1)              # simple temporal average pooling
        h = self.dropout(h)
        return self.fc(h)


# ==========================
# Training helpers
# ==========================

def make_model(cfg: Config) -> nn.Module:
    if cfg.model == 'lstm':
        return LSTMClassifier(cfg.input_dim, cfg.hidden_dim, cfg.num_layers, cfg.num_classes,
                              dropout=cfg.dropout, bidirectional=cfg.bidirectional)
    elif cfg.model == 'cnn':
        return CNN1DClassifier(cfg.input_dim, cfg.num_classes, hidden=cfg.hidden_dim, dropout=cfg.dropout)
    elif cfg.model == 'transformer':
        return TransformerEncoderClassifier(cfg.input_dim, cfg.num_classes, d_model=cfg.hidden_dim,
                                            nhead=cfg.num_heads, num_layers=cfg.num_layers,
                                            dim_feedforward=cfg.ff_dim, dropout=cfg.dropout)
    else:
        raise ValueError(f"Unknown model: {cfg.model}")


def epoch_step(model, loader, criterion, device, optimizer: Optional[torch.optim.Optimizer] = None):
    is_train = optimizer is not None
    model.train(is_train)
    all_logits, all_targets = [], []
    total_loss = 0.0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        if is_train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item() * X.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    preds = logits.argmax(dim=1).numpy()
    y_true = targets.numpy()
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average='macro')
    return avg_loss, acc, f1


def train_model(model, cfg: Config, train_ds: Dataset, val_ds: Dataset):
    device = cfg.device
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    #model = make_model(cfg).to(device)

    # Class weights example (uniform here; replace with real imbalance-aware weights if needed)
    class_weights = torch.ones(cfg.num_classes, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val = float('inf')
    patience = cfg.early_stop_patience
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc, train_f1 = epoch_step(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc, val_f1 = epoch_step(model, val_loader, criterion, device, optimizer=None)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | Train Loss {train_loss:.4f} Acc {train_acc:.3f} F1 {train_f1:.3f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.3f} F1 {val_f1:.3f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'cfg': cfg.__dict__,
            }, cfg.save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

    # Load best
    if os.path.exists(cfg.save_path):
        ckpt = torch.load(cfg.save_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    return model


# ==========================
# Main
# ==========================

def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='lstm', choices=['lstm', 'cnn', 'transformer'])
    p.add_argument('--input_dim', type=int, default=3)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--bidirectional', action='store_true')

    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--ff_dim', type=int, default=256)

    p.add_argument('--num_classes', type=int, default=3)
    p.add_argument('--seq_len', type=int, default=150)

    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--early_stop_patience', type=int, default=8)
    p.add_argument('--save_path', type=str, default='checkpoint.pt')
    args = p.parse_args(args=None if __name__ == "__main__" else [])

    cfg = Config(
        model=args.model,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_classes=args.num_classes,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        save_path=args.save_path,
    )
    return cfg


def main():
    set_seed(7)
    cfg = parse_args()
    cfg.input_dim = 6 #we have 6 features ph,ec,moisture,yaw,pitch and roll
    cfg.num_classes = 2


    # Create synthetic dataset you can immediately train on
    N, T, D, C = 3000, cfg.seq_len, cfg.input_dim, cfg.num_classes
    #X, y = make_synthetic_dataset(N, T, D, C)
    #ds = TimeSeriesDataset(X, y)

    # Training on dataset where there is a sinkhole formation
    ds = CSVDataset("C://Users//Ritika//Ritika//Github//SH//data//ball_train_sinkhole.csv", seq_len=30)
    val_ds = CSVDataset("C://Users//Ritika//Ritika//Github//SH//data//ball_train_sinkhole.csv", seq_len=30)

    # Train/val split
    #val_frac = 0.2
    #val_size = int(len(ds) * val_frac) # 1*0.2=0
    #train_size = len(ds) - val_size #1-0=1
    #train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_ds = ds

    print(f"Training {cfg.model.upper()} on synthetic data: N={N}, T={T}, D={D}, Classes={C}")
    device = cfg.device
    model = make_model(cfg).to(device)
    model = train_model(model, cfg, train_ds, val_ds)
    model = train_model(model, cfg, train_ds, val_ds)
    model = train_model(model, cfg, train_ds, val_ds)
    ds = CSVDataset("C://Users//Ritika//Ritika//Github//SH//data//ball_train_sinkhole_2.csv", seq_len=20)
    val_ds = CSVDataset("C://Users//Ritika//Ritika//Github//SH//data//ball_train_sinkhole_2.csv", seq_len=20)
    train_ds = ds
    model = train_model(model, cfg, train_ds, val_ds)


    # Training on dataset where there is no sinkhole formation
    ds = CSVDataset("C://Users//Ritika//Ritika//Github//SH//data//ball_train_no_sinkhole.csv", seq_len=20)
    val_ds = CSVDataset("C://Users//Ritika//Ritika//Github//SH//data//ball_train_no_sinkhole.csv", seq_len=20)
    train_ds = ds
    model = train_model(model, cfg, train_ds, val_ds)
    ds = CSVDataset("C://Users//Ritika//Ritika//Github//SH//data//ball_train_no_sinkhole_2.csv", seq_len=10)
    val_ds = CSVDataset("C://Users//Ritika//Ritika//Github//SH//data//ball_train_no_sinkhole_2.csv", seq_len=10)
    train_ds = ds
    model = train_model(model, cfg, train_ds, val_ds)
    #model = train_model(model, cfg, train_ds, val_ds)
    #model = train_model(model, cfg, train_ds, val_ds)



    # Quick evaluation on validation set
    device = cfg.device
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc, val_f1 = epoch_step(model.to(device), val_loader, criterion, device, optimizer=None)
    print(f"\nBest model validation — Loss: {val_loss:.4f} | Acc: {val_acc:.3f} | F1: {val_f1:.3f}")

    # Example: exporting the trained model
    example = torch.randn(1, cfg.seq_len, cfg.input_dim).to(device)
    torch.jit.trace(model, example).save("ts_classifier_traced.pt")
    print("Saved TorchScript model to ts_classifier_traced.pt")

    #Start : Test the model with a test csv file
    #1. Positive sinkhole test
    new_ds = CSVDataset("C://Users//Ritika//Ritika//Github//SH//data//ball_test_sinkhole.csv", seq_len=30)
    new_loader = torch.utils.data.DataLoader(new_ds, batch_size=1, shuffle=False)
    with torch.no_grad():
        for X, _ in new_loader:
            X = X.to(cfg.device)
            probs = torch.softmax(model(X), dim=1)
            pred_class = probs.argmax(dim=1).item()
            print("Predicted label:", pred_class, "| Probabilities:", probs.cpu().numpy())
    #test_ds_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    # model.eval() # Set the model to evaluation mode
    # for X, y in test_ds_loader:
    #     #X = X.to(device)
    #     logits = model(X)
    new_ds = CSVDataset("C://Users//Ritika//Ritika//Github//SH//data//ball_test_no_sinkhole.csv", seq_len=10)
    new_loader = torch.utils.data.DataLoader(new_ds, batch_size=1, shuffle=False)
    with torch.no_grad():
        for X, _ in new_loader:
            X = X.to(cfg.device)
            probs = torch.softmax(model(X), dim=1)
            pred_class = probs.argmax(dim=1).item()
            print("Predicted label:", pred_class, "| Probabilities:", probs.cpu().numpy())

    #End



if __name__ == "__main__":
    main()
