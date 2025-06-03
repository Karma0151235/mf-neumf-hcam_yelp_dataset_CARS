# --- run_mf.py ---

import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from utils.data_loader import load_yelp, preprocess
from utils.splitter import random_split_3way, split_time
from utils.evaluate import evaluate_model
from utils.visualize import plot_fold_losses, plot_train_vs_test
from models.mf import MF


def train_mf(train_df, val_df, num_users, num_items, latent_dim=32, epochs=20, batch_size=128, lr=0.02):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = MF(num_users, num_items, latent_dim).to(device)

    users_train = torch.tensor(train_df['user'].values, dtype=torch.long)
    items_train = torch.tensor(train_df['item'].values, dtype=torch.long)
    ratings_train = torch.tensor(train_df['rating'].values, dtype=torch.float)

    users_val = torch.tensor(val_df['user'].values, dtype=torch.long)
    items_val = torch.tensor(val_df['item'].values, dtype=torch.long)
    ratings_val = torch.tensor(val_df['rating'].values, dtype=torch.float)

    train_loader = DataLoader(TensorDataset(users_train, items_train, ratings_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(users_val, items_val, ratings_val), batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for u, i, r in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            pred = model(u, i)
            loss = loss_fn(pred, r)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for u, i, r in val_loader:
                u, i, r = u.to(device), i.to(device), r.to(device)
                pred = model(u, i)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(r.cpu().numpy())
                val_loss += loss_fn(pred, r).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Final metrics
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    val_rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
    val_mae = np.mean(np.abs(val_preds - val_targets))

    model.eval()
    train_preds, train_targets = [], []
    with torch.no_grad():
        for u, i, r in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            pred = model(u, i)
            train_preds.extend(pred.cpu().numpy())
            train_targets.extend(r.cpu().numpy())

    train_preds = np.array(train_preds)
    train_targets = np.array(train_targets)
    train_rmse = np.sqrt(np.mean((train_preds - train_targets) ** 2))
    train_mae = np.mean(np.abs(train_preds - train_targets))

    print(f"Final Val RMSE: {val_rmse:.4f}, Final Val MAE: {val_mae:.4f}")
    print(f"RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")

    return model, train_losses, val_losses


if __name__ == '__main__':
    path_json_dir = 'datasets/'
    raw_df = load_yelp(path_json_dir, sample_size=500000)
    df, _ = preprocess(raw_df, min_uc=3, min_ic=3)

    num_users = df['user'].nunique()
    num_items = df['item'].nunique()
    print(f"Dataset Size: {len(df)}, Users: {num_users}, Items: {num_items}")

    print("\n--- 10x Random 80/10/10 Splits ---")
    fold_rmse, fold_mae, fold_losses, fold_val_losses = [], [], [], []
    for seed in range(10):
        print(f"\n=== Split {seed + 1}/10 ===")
        train_df, val_df, test_df = random_split_3way(df, seed)
        model, train_losses, val_losses = train_mf(train_df, val_df, num_users, num_items)
        rmse, mae = evaluate_model(model, test_df)
        fold_rmse.append(rmse)
        fold_mae.append(mae)
        fold_losses.append(train_losses)
        fold_val_losses.append(val_losses)

    print("\n=== Average Results over 10 Splits ===")
    print(f"Avg RMSE: {np.mean(fold_rmse):.4f}, Avg MAE: {np.mean(fold_mae):.4f}")

    plot_fold_losses(fold_losses, title="MF Training Loss")
    plot_train_vs_test(fold_losses, fold_val_losses, title="MF Train vs Validation Loss")

    print("\n--- Time-Based Split (80/20) ---")
    train_df, test_df = split_time(df, ratio=0.8)
    val_df = train_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    train_df = train_df.drop(index=val_df.index).reset_index(drop=True)

    model, _, _ = train_mf(train_df, val_df, num_users, num_items)
    rmse, mae = evaluate_model(model, test_df)
    print(f"Time-based Split Results — RMSE: {rmse:.4f}, MAE: {mae:.4f}")
