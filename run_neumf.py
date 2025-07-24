import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from utils.data_loader import load_yelp, preprocess
from utils.splitter import random_split_3way, split_time
from utils.evaluate import evaluate_model
from utils.visualize import plot_train_vs_test
from models.neumf import NeuMF

import warnings
warnings.filterwarnings('ignore')
print("This Script Ignores All Warnings")


def train_neumf(train_df, val_df, num_users, num_items, latent_dim_mf=16, latent_dim_mlp=16, mlp_layers=[32, 16, 8], dropout=0.31778351694762974, epochs=7, batch_size=128, lr=0.00043885100849082396):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = NeuMF(num_users, num_items, latent_dim_mf, latent_dim_mlp, mlp_layers, dropout).to(device)
    model.device = device  # Ensure model.device is set

    train_loader = DataLoader(TensorDataset(
        torch.tensor(train_df['user'].values, dtype=torch.long),
        torch.tensor(train_df['item'].values, dtype=torch.long),
        torch.tensor(train_df['rating'].values, dtype=torch.float)
    ), batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(TensorDataset(
        torch.tensor(val_df['user'].values, dtype=torch.long),
        torch.tensor(val_df['item'].values, dtype=torch.long),
        torch.tensor(val_df['rating'].values, dtype=torch.float)
    ), batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    train_maes, val_maes = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for u, i, r in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            pred = model(u, i).view(-1)
            loss = loss_fn(pred, r)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_preds, val_targets, val_loss = [], [], 0
        with torch.no_grad():
            for u, i, r in val_loader:
                u, i, r = u.to(device), i.to(device), r.to(device)
                pred = model(u, i).view(-1)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(r.cpu().numpy())
                val_loss += loss_fn(pred, r).item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Train metrics
        train_preds, train_targets = [], []
        with torch.no_grad():
            for u, i, r in train_loader:
                u, i, r = u.to(device), i.to(device), r.to(device)
                pred = model(u, i).view(-1)
                train_preds.extend(pred.cpu().numpy())
                train_targets.extend(r.cpu().numpy())

        val_mae = mean_absolute_error(val_targets, val_preds)
        train_mae = mean_absolute_error(train_targets, train_preds)
        val_maes.append(val_mae)
        train_maes.append(train_mae)

        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")

    return model, train_losses, val_losses, train_maes, val_maes


def run_neumf(df, time_based=False):
    num_users = df['user'].nunique()
    num_items = df['item'].nunique()

    fold_rmse, fold_mae, fold_losses, fold_val_losses = [], [], [], []
    fold_train_maes, fold_val_maes = [], []

    if time_based:
        print("\n--- Time-Based Split (80/20) ---")
        train_df, test_df = split_time(df, ratio=0.8)
        val_df = train_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
        train_df = train_df.drop(index=val_df.index).reset_index(drop=True)

        model, train_losses, val_losses, train_maes, val_maes = train_neumf(train_df, val_df, num_users, num_items)
        rmse, mae = evaluate_model(model, test_df)

        # Final train set metrics
        train_loader = DataLoader(TensorDataset(
            torch.tensor(train_df['user'].values),
            torch.tensor(train_df['item'].values),
            torch.tensor(train_df['rating'].values)
        ), batch_size=256)
        train_preds, train_targets = [], []
        model.eval()
        with torch.no_grad():
            for u, i, r in train_loader:
                u, i = u.to(model.device), i.to(model.device)
                pred = model(u, i).view(-1)
                train_preds.extend(pred.cpu().numpy())
                train_targets.extend(r.numpy())
        train_mae = mean_absolute_error(train_targets, train_preds)
        train_rmse = mean_squared_error(train_targets, train_preds, squared=False)

        print(f"Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}")
        print(f"Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}")

        return {
            'rmse': rmse,
            'mae': mae,
            'train_losses': [train_losses],
            'val_losses': [val_losses],
            'train_maes': [train_maes],
            'val_maes': [val_maes]
        }

    print("\n--- 10x Random 80/10/10 Splits ---")
    for seed in range(10):
        print(f"\n=== NeuMF Split {seed + 1}/10 ===")
        train_df, val_df, test_df = random_split_3way(df, seed)
        model, train_losses, val_losses, train_maes, val_maes = train_neumf(train_df, val_df, num_users, num_items)
        rmse, mae = evaluate_model(model, test_df)

        # Final train metrics
        train_loader = DataLoader(TensorDataset(
            torch.tensor(train_df['user'].values),
            torch.tensor(train_df['item'].values),
            torch.tensor(train_df['rating'].values)
        ), batch_size=256)
        train_preds, train_targets = [], []
        model.eval()
        with torch.no_grad():
            for u, i, r in train_loader:
                u, i = u.to(model.device), i.to(model.device)
                pred = model(u, i).view(-1)
                train_preds.extend(pred.cpu().numpy())
                train_targets.extend(r.numpy())
        train_mae = mean_absolute_error(train_targets, train_preds)
        train_rmse = mean_squared_error(train_targets, train_preds, squared=False)

        print(f"Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}")
        print(f"Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}")

        fold_rmse.append(rmse)
        fold_mae.append(mae)
        fold_losses.append(train_losses)
        fold_val_losses.append(val_losses)
        fold_train_maes.append(train_maes)
        fold_val_maes.append(val_maes)

    print("\n=== Average Results over 10 Splits ===")
    print(f"Avg RMSE: {np.mean(fold_rmse):.4f}, Avg MAE: {np.mean(fold_mae):.4f}")

    return {
        'rmse': np.mean(fold_rmse),
        'mae': np.mean(fold_mae),
        'train_losses': fold_losses,
        'val_losses': fold_val_losses,
        'train_maes': fold_train_maes,
        'val_maes': fold_val_maes
    }


if __name__ == '__main__':
    path_json_dir = 'datasets/'
    raw_df = load_yelp(path_json_dir, sample_size=500000)
    df, _ = preprocess(raw_df, min_uc=3, min_ic=3)

    results_tb = run_neumf(df, time_based=True)
    plot_train_vs_test(results_tb['train_losses'], results_tb['val_losses'], "NeuMF Train vs Val Loss (Time-Based)")
    results_cv = run_neumf(df)
    plot_train_vs_test(results_cv['train_losses'], results_cv['val_losses'], "NeuMF Train vs Val Loss (10-fold)")
