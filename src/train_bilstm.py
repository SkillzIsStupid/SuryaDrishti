#!/usr/bin/env python3
"""
train_bilstm.py

Train a Bidirectional LSTM (BiLSTM) for solar irradiance forecasting.
Designed to run on Mac GPU (MPS) if available, or CUDA/CPU.

Usage:
    python src/train_bilstm.py --epochs 40 --batch 64 --lr 0.005 --hidden 128 --val-split 0.15

Outputs:
    - models/bilstm_best.pth
    - models/bilstm_last.pth
    - logs/train_history.csv
"""

import os
import argparse
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# --------------------------
# Utilities
# --------------------------
def pick_device(prefer: str = None):
    """Pick device: cuda -> mps -> cpu"""
    if prefer:
        d = torch.device(prefer)
        return d
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS (Apple Silicon)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------
# Model
# --------------------------
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, bidirectional=True, out_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_size * self.num_directions, out_size)
        self.act = nn.Sigmoid()  # outputs 0..1 to match normalized targets

        # weight init
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.constant_(p, 0.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)  # out: (B, T, hidden * num_directions)
        # take last time-step
        last = out[:, -1, :]  # (B, hidden * num_directions)
        out = self.fc(last)
        out = self.act(out)
        return out


# --------------------------
# Training functions
# --------------------------
def load_data(x_path="data/processed/X.npy", y_path="data/processed/y.npy"):
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Processed files not found: {x_path} {y_path}")
    X = np.load(x_path)
    y = np.load(y_path)
    return X, y


def create_loaders(X, y, batch_size=64, val_split=0.15, seed=42):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(-1))
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
    return (total_loss / n) if n > 0 else float("nan")


def train_loop(model, train_loader, val_loader, optimizer, criterion, scheduler, device, epochs, clip, out_dir, patience=8):
    best_val = float("inf")
    best_epoch = -1
    history = []

    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "bilstm_best.pth")
    last_path = os.path.join(out_dir, "bilstm_last.pth")

    no_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            # grad clipping
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)
            n += xb.size(0)

        avg_train_loss = epoch_loss / n if n > 0 else float("nan")
        val_loss = validate(model, val_loader, criterion, device)

        # scheduler step
        if scheduler is not None:
            scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{epochs} - train_loss: {avg_train_loss:.6e} val_loss: {val_loss:.6e} time: {elapsed:.1f}s")

        history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": val_loss, "time_s": elapsed})

        # checkpoint
        torch.save(model.state_dict(), last_path)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            no_improve = 0
            print(f"  -> New best model saved (epoch {epoch}, val_loss {val_loss:.6e})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping (no improvement for {patience} epochs).")
                break

    # final save
    torch.save(model.state_dict(), last_path)
    hist_df = pd.DataFrame(history)
    hist_path = os.path.join("logs", "train_history.csv")
    os.makedirs("logs", exist_ok=True)
    hist_df.to_csv(hist_path, index=False)
    print(f"Training finished. Best epoch: {best_epoch} val_loss: {best_val:.6e}")
    print(f"Saved best -> {best_path}, last -> {last_path}, history -> {hist_path}")
    return hist_df


# --------------------------
# CLI entry
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM for solar irradiance prediction")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--clip", type=float, default=3.0)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--device", type=str, default=None, help="cuda|mps|cpu (auto-detect if not set)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="models")
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    print("Using device:", device)

    # Load data
    X, y = load_data()
    print("Loaded processed data shapes:", X.shape, y.shape)
    print("Overall stats -> X mean/std:", X.mean(), X.std(), "y mean/std:", y.mean(), y.std())

    input_size = X.shape[2]
    model = BiLSTMModel(input_size=input_size, hidden_size=args.hidden, num_layers=args.layers, dropout=args.dropout, bidirectional=True)
    model = model.to(device)

    train_loader, val_loader = create_loaders(X, y, batch_size=args.batch, val_split=args.val_split, seed=args.seed)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # ReduceLROnPlateau monitors val loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5, verbose=True)

    hist = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        clip=args.clip,
        out_dir=args.out,
        patience=args.patience,
    )

    # Save normalization meta if available (help predict.py)
    norm_path = Path("models/norm.npy")
    if not norm_path.exists():
        print("No models/norm.npy found — ensuring prediction step has normalization content.")
    else:
        print("Normalization file exists:", norm_path)

if __name__ == "__main__":
    main()
