import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from utils.data_loader import load_yelp, preprocess
from utils.splitter import random_split_3way, split_time
from utils.evaluate import evaluate_model
from utils.visualize import plot_train_vs_test
from models.hcamneumf import HCAMNeuMF

import warnings
warnings.filterwarnings('ignore')
print("This Script Ignores All Warnings")


def train_hcamneumf(train_data, val_data, num_users, num_items, context_dim, latent_dim_mf=128, latent_dim_mlp=128, mlp_layers=[256, 128, 64, 32, 16, 8], dropout=0.3483810504472348, epochs=10, batch_size=128, lr=0.0001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = HCAMNeuMF(num_users, num_items, context_dim, latent_dim_mf, latent_dim_mlp, mlp_layers, dropout).to(device)
    model.device = device

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    train_maes, val_maes = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for u, i, c, r in train_loader:
            u, i, c, r = u.to(device), i.to(device), c.to(device), r.to(device)
            pred = model(u, i, c).view(-1)
            loss = loss_fn(pred, r)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0
        with torch.no_grad():
            for u, i, c, r in val_loader:
                u, i, c, r = u.to(device), i.to(device), c.to(device), r.to(device)
                pred = model(u, i, c).view(-1)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(r.cpu().numpy())
                val_loss += loss_fn(pred, r).item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        train_preds, train_targets = [], []
        with torch.no_grad():
            for u, i, c, r in train_loader:
                u, i, c, r = u.to(device), i.to(device), c.to(device), r.to(device)
                pred = model(u, i, c).view(-1)
                train_preds.extend(pred.cpu().numpy())
                train_targets.extend(r.cpu().numpy())

        train_mae = mean_absolute_error(train_targets, train_preds)
        val_mae = mean_absolute_error(val_targets, val_preds)
        train_maes.append(train_mae)
        val_maes.append(val_mae)

        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")

    return model, train_losses, val_losses, train_maes, val_maes


def run_hcamneumf(df, context_matrix, time_based=False):
    df = df.reset_index(drop=True)
    context_matrix = np.array(context_matrix, dtype=np.float32)  # <--- ensure valid format
    df['context'] = list(context_matrix)

    num_users = df['user'].nunique()
    num_items = df['item'].nunique()
    context_dim = context_matrix.shape[1]

    def make_dataset(split_df):
        users = torch.tensor(split_df['user'].values, dtype=torch.long)
        items = torch.tensor(split_df['item'].values, dtype=torch.long)
        contexts = torch.tensor(np.stack(split_df['context'].values), dtype=torch.float32)
        ratings = torch.tensor(split_df['rating'].values, dtype=torch.float32)
        return TensorDataset(users, items, contexts, ratings)

    fold_rmse, fold_mae = [], []
    fold_losses, fold_val_losses = [], []
    fold_train_maes, fold_val_maes = [], []

    if time_based:
        print("\n--- Time-Based Split (80/20) ---")
        train_df, test_df = split_time(df, ratio=0.8)
        val_df = train_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
        train_df = train_df.drop(index=val_df.index).reset_index(drop=True)

        train_data = make_dataset(train_df)
        val_data = make_dataset(val_df)
        test_data = make_dataset(test_df)

        model, train_losses, val_losses, train_maes, val_maes = train_hcamneumf(
            train_data, val_data, num_users, num_items, context_dim)

        context_test = torch.tensor(np.stack(test_df['context'].values), dtype=torch.float32)
        rmse, mae = evaluate_model(model, test_df, context_tensor=context_test)

        train_loader = DataLoader(train_data, batch_size=128)
        train_preds, train_targets = [], []
        with torch.no_grad():
            for u, i, c, r in train_loader:
                u, i, c = u.to(model.device), i.to(model.device), c.to(model.device)
                pred = model(u, i, c).view(-1)
                train_preds.extend(pred.cpu().numpy())
                train_targets.extend(r.numpy())

        train_mae_final = mean_absolute_error(train_targets, train_preds)
        train_rmse = mean_squared_error(train_targets, train_preds, squared=False)

        print(f"Train MAE: {train_mae_final:.4f}, Train RMSE: {train_rmse:.4f}")
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
        print(f"\n=== HCAM Split {seed + 1}/10 ===")
        train_df, val_df, test_df = random_split_3way(df, seed)

        train_data = make_dataset(train_df)
        val_data = make_dataset(val_df)
        test_data = make_dataset(test_df)

        model, train_losses, val_losses, train_maes, val_maes = train_hcamneumf(
            train_data, val_data, num_users, num_items, context_dim)

        context_test = torch.tensor(np.stack(test_df['context'].values), dtype=torch.float32)
        rmse, mae = evaluate_model(model, test_df, context_tensor=context_test)

        train_loader = DataLoader(train_data, batch_size=128)
        train_preds, train_targets = [], []
        with torch.no_grad():
            for u, i, c, r in train_loader:
                u, i, c = u.to(model.device), i.to(model.device), c.to(model.device)
                pred = model(u, i, c).view(-1)
                train_preds.extend(pred.cpu().numpy())
                train_targets.extend(r.numpy())

        train_mae_final = mean_absolute_error(train_targets, train_preds)
        train_rmse = mean_squared_error(train_targets, train_preds, squared=False)

        print(f"Train MAE: {train_mae_final:.4f}, Train RMSE: {train_rmse:.4f}")
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
    df, context_matrix = preprocess(raw_df, min_uc=3, min_ic=3)

    hcam_tb_results = run_hcamneumf(df, context_matrix, time_based=True)
    plot_train_vs_test(hcam_tb_results['train_losses'], hcam_tb_results['val_losses'], "HCAM-NeuMF Train vs Val Loss (Time-Based)")

    hcam_results = run_hcamneumf(df, context_matrix)
    plot_train_vs_test(hcam_results['train_losses'], hcam_results['val_losses'], "HCAM-NeuMF Train vs Val Loss (10-fold)")
