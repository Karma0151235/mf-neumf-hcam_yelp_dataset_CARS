import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from models.hcamneumf import HCAMNeuMF
from utils.data_loader import load_yelp, preprocess
from utils.splitter import random_split_3way

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load and prepare data
path_json_dir = 'datasets/'
raw_df = load_yelp(path_json_dir, sample_size=500_000)
df, context_matrix = preprocess(raw_df, min_uc=3, min_ic=3)
df = df.reset_index(drop=True)
df['context'] = [row.astype(np.float32) for row in context_matrix]
context_dim = context_matrix.shape[1]
num_users = df['user'].nunique()
num_items = df['item'].nunique()

# Prepare a fixed split for optimization
train_df, val_df, _ = random_split_3way(df, seed=42)

def make_dataset(split_df):
    users = torch.tensor(split_df['user'].values, dtype=torch.long)
    items = torch.tensor(split_df['item'].values, dtype=torch.long)
    contexts = torch.tensor(np.stack(split_df['context'].values), dtype=torch.float32)
    ratings = torch.tensor(split_df['rating'].values, dtype=torch.float32)
    return TensorDataset(users, items, contexts, ratings)

train_data = make_dataset(train_df)
val_data = make_dataset(val_df)

# Hyperparameter space
space = [
    Integer(16, 128, name='latent_dim_mf'),
    Integer(16, 128, name='latent_dim_mlp'),
    Real(0.0001, 0.01, prior='log-uniform', name='lr'),
    Real(0.0, 0.7, name='dropout')
]

@use_named_args(space)
def objective(latent_dim_mf, latent_dim_mlp, lr, dropout):
    mlp_layers = [latent_dim_mlp * 2, latent_dim_mlp, latent_dim_mlp // 2]

    model = HCAMNeuMF(
        num_users=num_users,
        num_items=num_items,
        context_dim=context_dim,
        latent_dim_mf=latent_dim_mf,
        latent_dim_mlp=latent_dim_mlp,
        mlp_layers=mlp_layers,
        dropout=dropout
    ).to(device)

    model.device = device
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128)

    # Train for a few epochs only
    for epoch in range(5):
        model.train()
        for u, i, c, r in train_loader:
            u, i, c, r = u.to(device), i.to(device), c.to(device), r.to(device)
            pred = model(u, i, c).view(-1)
            loss = loss_fn(pred, r)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for u, i, c, r in val_loader:
            u, i, c, r = u.to(device), i.to(device), c.to(device), r.to(device)
            pred = model(u, i, c).view(-1)
            val_preds.extend(pred.cpu().numpy())
            val_targets.extend(r.cpu().numpy())

    rmse = mean_squared_error(val_targets, val_preds, squared=False)
    print(f"Tested: mf={latent_dim_mf}, mlp={latent_dim_mlp}, dropout={dropout:.4f}, lr={lr:.5f} => RMSE: {rmse:.4f}")
    return rmse

# Run optimization
print("Optimizing HCAM-NeuMF...")
result = gp_minimize(objective, space, n_calls=30, random_state=42, verbose=True)

# Print best
print("\nBest Parameters:")
print(f"latent_dim_mf: {result.x[0]}")
print(f"latent_dim_mlp: {result.x[1]}")
print(f"lr: {result.x[2]}")
print(f"dropout: {result.x[3]}")
