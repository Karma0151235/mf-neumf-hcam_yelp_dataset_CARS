import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.neumf import NeuMF
from utils.data_loader import load_yelp, preprocess
from utils.splitter import random_split_3way
from utils.evaluate import evaluate_model

# Load and preprocess data
path_json_dir = 'datasets/'
raw_df = load_yelp(path_json_dir, sample_size=500000)
df, _ = preprocess(raw_df, min_uc=3, min_ic=3)

train_df, val_df, _ = random_split_3way(df, seed=42)
num_users = df['user'].nunique()
num_items = df['item'].nunique()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare loaders
def make_loader(df, batch_size):
    return DataLoader(TensorDataset(
        torch.tensor(df['user'].values, dtype=torch.long),
        torch.tensor(df['item'].values, dtype=torch.long),
        torch.tensor(df['rating'].values, dtype=torch.float)
    ), batch_size=batch_size, shuffle=True)

# Define the hyperparameter search space
space = [
    Integer(8, 64, name='latent_dim_mf'),
    Integer(8, 64, name='latent_dim_mlp'),
    Real(0.0001, 0.05, prior='log-uniform', name='lr'),
    Real(0.0, 0.6, name='dropout')
]

@use_named_args(space)
def objective(latent_dim_mf, latent_dim_mlp, lr, dropout):
    model = NeuMF(
        num_users=num_users,
        num_items=num_items,
        latent_dim_mf=latent_dim_mf,
        latent_dim_mlp=latent_dim_mlp,
        mlp_layers=[latent_dim_mlp * 2, 16, 8],
        dropout=dropout
    ).to(device)
    model.device = device

    train_loader = make_loader(train_df, batch_size=128)
    val_loader = make_loader(val_df, batch_size=128)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(10):
        for u, i, r in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            pred = model(u, i).view(-1)
            loss = loss_fn(pred, r)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate on validation set
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for u, i, r in val_loader:
            u, i = u.to(device), i.to(device)
            pred = model(u, i).view(-1)
            val_preds.extend(pred.cpu().numpy())
            val_targets.extend(r.numpy())

    val_rmse = mean_squared_error(val_targets, val_preds, squared=False)
    print(f"Params: mf={latent_dim_mf}, mlp={latent_dim_mlp}, lr={lr:.5f}, dropout={dropout:.2f} | Val RMSE: {val_rmse:.4f}")
    return val_rmse

# Run optimization
print("Running Bayesian Optimization for NeuMF...")
result = gp_minimize(objective, space, n_calls=30, random_state=42, verbose=True)

print("\nBest parameters found:")
print(f"latent_dim_mf: {result.x[0]}")
print(f"latent_dim_mlp: {result.x[1]}")
print(f"lr: {result.x[2]}")
print(f"dropout: {result.x[3]}")
