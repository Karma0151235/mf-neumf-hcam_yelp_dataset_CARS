import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from utils.data_loader import load_yelp, preprocess
from utils.splitter import random_split_3way, split_time
from utils.evaluate import evaluate_model
from utils.visualize import plot_fold_losses, plot_train_vs_test
from models.mf import MF

import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings
warnings.warn("This warning will be hidden")
print("This Script Ignores All Warnings")

def train_mf(train_df, val_df, num_users, num_items, latent_dim=8, epochs=10, batch_size=95, lr=0.01863032884378607):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = MF(num_users, num_items, latent_dim).to(device)

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
            pred = model(u, i)
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
                pred = model(u, i)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(r.cpu().numpy())
                val_loss += loss_fn(pred, r).item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_rmse = mean_squared_error(val_targets, val_preds, squared=False)
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_maes.append(val_mae)

        # Also compute train metrics
        train_preds, train_targets = [], []
        with torch.no_grad():
            for u, i, r in train_loader:
                u, i, r = u.to(device), i.to(device), r.to(device)
                pred = model(u, i)
                train_preds.extend(pred.cpu().numpy())
                train_targets.extend(r.cpu().numpy())
        train_rmse = mean_squared_error(train_targets, train_preds, squared=False)
        train_mae = mean_absolute_error(train_targets, train_preds)
        train_maes.append(train_mae)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")

    return model, train_losses, val_losses, train_maes, val_maes

def run_mf(df, time_based=False):
    num_users = df['user'].nunique()
    num_items = df['item'].nunique()

    fold_rmse, fold_mae, fold_losses, fold_val_losses = [], [], [], []
    fold_train_maes, fold_val_maes = [], []

    if time_based:
        print("\n--- Time-Based Split (80/20) ---")
        train_df, test_df = split_time(df, ratio=0.8)
        val_df = train_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
        train_df = train_df.drop(index=val_df.index).reset_index(drop=True)

        model, train_losses, val_losses, train_maes, val_maes = train_mf(train_df, val_df, num_users, num_items)
        rmse, mae = evaluate_model(model, test_df)

        # Train set metrics
        train_preds, train_targets = [], []
        train_loader = DataLoader(TensorDataset(
            torch.tensor(train_df['user'].values, dtype=torch.long),
            torch.tensor(train_df['item'].values, dtype=torch.long),
            torch.tensor(train_df['rating'].values, dtype=torch.float)
        ), batch_size=128)
        with torch.no_grad():
            for u, i, r in train_loader:
                pred = model(u, i)
                train_preds.extend(pred.cpu().numpy())
                train_targets.extend(r.cpu().numpy())
        train_rmse = mean_squared_error(train_targets, train_preds, squared=False)
        train_mae = mean_absolute_error(train_targets, train_preds)

        print(f"Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}")
        print(f"Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}")

        return {
            'rmse': rmse,
            'mae': mae,
            'train_losses': [train_losses],
            'val_losses': [val_losses],
            'train_maes': [train_maes],
            'val_maes': [val_maes]
        }

    else:
        print("\n--- 10x Random 80/10/10 Splits ---")
        for seed in range(10):
            print(f"\n=== Split {seed + 1}/10 ===")
            train_df, val_df, test_df = random_split_3way(df, seed)
            model, train_losses, val_losses, train_maes, val_maes = train_mf(train_df, val_df, num_users, num_items)
            rmse, mae = evaluate_model(model, test_df)

            # Train set metrics
            train_loader = DataLoader(TensorDataset(
                torch.tensor(train_df['user'].values, dtype=torch.long),
                torch.tensor(train_df['item'].values, dtype=torch.long),
                torch.tensor(train_df['rating'].values, dtype=torch.float)
            ), batch_size=128)
            train_preds, train_targets = [], []
            with torch.no_grad():
                for u, i, r in train_loader:
                    pred = model(u, i)
                    train_preds.extend(pred.cpu().numpy())
                    train_targets.extend(r.cpu().numpy())
            train_rmse = mean_squared_error(train_targets, train_preds, squared=False)
            train_mae = mean_absolute_error(train_targets, train_preds)

            print(f"Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}")
            print(f"Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}")

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

    mf_tb_results = run_mf(df, time_based=True)
    plot_train_vs_test(mf_tb_results['train_losses'], mf_tb_results['val_losses'], "MF Train vs Val Loss (Time-Based)")
    mf_results = run_mf(df)
    plot_train_vs_test(mf_results['train_losses'], mf_results['val_losses'], "MF Train vs Val Loss (10-fold)")
