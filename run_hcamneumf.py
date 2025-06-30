import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from utils.data_loader import load_yelp, preprocess
from utils.splitter import random_split_3way, split_time
from utils.evaluate import evaluate_model
from utils.visualize import plot_fold_losses, plot_train_vs_test
from utils.autoencoder import Autoencoder
from models.hcamneumf import HCAMNeuMF

def load_encoded_context(encoded_path="data/context_latents.npy"):
    return np.load(encoded_path)

def train_hcamneumf(model, train_data, val_data, epochs=20, batch_size=128, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = model.to(device)
    model.device = device

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for u, i, c, r in train_loader:
            u, i, c, r = u.to(device), i.to(device), c.to(device), r.to(device)
            pred = model(u, i, c).squeeze()
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
            for u, i, c, r in val_loader:
                u, i, c, r = u.to(device), i.to(device), c.to(device), r.to(device)
                pred = model(u, i, c).squeeze()
                pred = pred.view(-1).cpu().numpy()
                r = r.view(-1).cpu().numpy()
                val_preds.extend(pred)
                val_targets.extend(r)
                val_loss += loss_fn(torch.tensor(pred), torch.tensor(r)).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return model, train_losses, val_losses

if __name__ == '__main__':
    path_json_dir = 'datasets/'
    raw_df = load_yelp(path_json_dir, sample_size=500000)
    df, context_matrix = preprocess(raw_df, min_uc=3, min_ic=3)
    context_latents = load_encoded_context()

    num_users = df['user'].nunique()
    num_items = df['item'].nunique()
    context_dim = context_latents.shape[1]

    df['context'] = list(context_latents)
    print(f"Dataset Size: {len(df)}, Users: {num_users}, Items: {num_items}, Context Dim: {context_dim}")

    print("\n--- 10x Random 80/10/10 Splits ---")
    fold_rmse, fold_mae, fold_losses, fold_val_losses = [], [], [], []
    for seed in range(10):
        print(f"\n=== Split {seed + 1}/10 ===")
        train_df, val_df, test_df = random_split_3way(df, seed)

        def make_dataset(split_df):
            users = torch.tensor(split_df['user'].values, dtype=torch.long)
            items = torch.tensor(split_df['item'].values, dtype=torch.long)
            contexts = torch.tensor(np.stack(split_df['context'].values), dtype=torch.float32)
            ratings = torch.tensor(split_df['rating'].values, dtype=torch.float32)
            return TensorDataset(users, items, contexts, ratings)

        train_data = make_dataset(train_df)
        val_data = make_dataset(val_df)
        test_data = make_dataset(test_df)

        model = HCAMNeuMF(
            num_users, num_items, context_dim,
            latent_dim_mf=16, latent_dim_mlp=32, mlp_layers=[64, 32, 16, 8]
        )

        model, train_losses, val_losses = train_hcamneumf(model, train_data, val_data)

        model.eval()
        test_preds, test_targets = [], []
        with torch.no_grad():
            for u, i, c, r in DataLoader(test_data, batch_size=128):
                u, i, c, r = u.to(model.device), i.to(model.device), c.to(model.device), r.to(model.device)
                pred = model(u, i, c).squeeze()
                test_preds.extend(pred.view(-1).cpu().numpy())
                test_targets.extend(r.view(-1).cpu().numpy())

        test_preds = np.array(test_preds)
        test_targets = np.array(test_targets)
        rmse = np.sqrt(np.mean((test_preds - test_targets) ** 2))
        mae = np.mean(np.abs(test_preds - test_targets))

        print(f"Final Val RMSE: {rmse:.4f}, Final Val MAE: {mae:.4f}")

        fold_rmse.append(rmse)
        fold_mae.append(mae)
        fold_losses.append(train_losses)
        fold_val_losses.append(val_losses)

    print("\n=== Average Results over 10 Splits ===")
    print(f"Avg RMSE: {np.mean(fold_rmse):.4f}, Avg MAE: {np.mean(fold_mae):.4f}")
    plot_fold_losses(fold_losses, title="HCAM-NeuMF Training Loss")
    plot_train_vs_test(fold_losses, fold_val_losses, title="HCAM-NeuMF Train vs Validation Loss")
