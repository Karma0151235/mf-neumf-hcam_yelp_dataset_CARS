import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_absolute_error

from utils.data_loader import load_yelp, preprocess
from models.mf import MF
from utils.evaluate import evaluate

# TRAINING

def train_mf(df, num_users, num_items, latent_dim=32, epochs=10, batch_size=128, lr=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    users = torch.tensor(df['user'].values, dtype=torch.long)
    items = torch.tensor(df['item'].values, dtype=torch.long)
    ratings = torch.tensor(df['rating'].values, dtype=torch.float)

    dataset = TensorDataset(users, items, ratings)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    model = MF(num_users, num_items, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

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
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    return model, val_loader

# RUN CODE

if __name__ == '__main__':
    path_json_dir = 'datasets/'
    raw_df = load_yelp(path_json_dir, sample_size=500000)
    df, _ = preprocess(raw_df, min_uc=3, min_ic=3)  # context returned but not used here

    num_users = df['user'].nunique()
    num_items = df['item'].nunique()
    print(f"Dataset Size: {len(df)}, Users: {num_users}, Items: {num_items}")

    model, val_loader = train_mf(df, num_users, num_items, latent_dim=32, epochs=10)
    evaluate(model, val_loader)