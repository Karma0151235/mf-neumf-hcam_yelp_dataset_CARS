import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from models.mf import MF
from utils.data_loader import load_yelp, preprocess

class MFTrainer:
    def __init__(self, train_data, val_data, num_users, num_items, device='cpu'):
        self.train_data = train_data
        self.val_data = val_data
        self.num_users = num_users
        self.num_items = num_items
        self.device = device

    def train_and_evaluate(self, latent_dim, lr, batch_size):
        model = MF(self.num_users, self.num_items, latent_dim=latent_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        model.train()
        for epoch in range(5):  # keep short for tuning
            for u, i, r in train_loader:
                u, i, r = u.to(self.device), i.to(self.device), r.to(self.device)
                pred = model(u, i)
                loss = loss_fn(pred, r)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        val_preds, val_targets = [], []
        val_loader = DataLoader(self.val_data, batch_size=256)
        with torch.no_grad():
            for u, i, r in val_loader:
                u, i = u.to(self.device), i.to(self.device)
                pred = model(u, i)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(r.numpy())

        return mean_absolute_error(val_targets, val_preds)

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_json_dir = 'datasets/'
raw_df = load_yelp(path_json_dir, sample_size=500000)
df, _ = preprocess(raw_df, min_uc=3, min_ic=3)

num_users = df['user'].nunique()
num_items = df['item'].nunique()

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_dataset = TensorDataset(
    torch.tensor(train_df['user'].values, dtype=torch.long),
    torch.tensor(train_df['item'].values, dtype=torch.long),
    torch.tensor(train_df['rating'].values, dtype=torch.float32)
)

val_dataset = TensorDataset(
    torch.tensor(val_df['user'].values, dtype=torch.long),
    torch.tensor(val_df['item'].values, dtype=torch.long),
    torch.tensor(val_df['rating'].values, dtype=torch.float32)
)

trainer = MFTrainer(train_dataset, val_dataset, num_users, num_items, device)

# === Run skopt ===
from skopt import gp_minimize

def objective(params):
    latent_dim, lr, batch_size = params
    mae = trainer.train_and_evaluate(latent_dim=int(latent_dim), lr=lr, batch_size=int(batch_size))
    print(f"Tested: latent_dim={latent_dim}, lr={lr}, batch_size={batch_size} -> MAE: {mae:.4f}")
    return mae

search_space = [
    Integer(8, 64, name='latent_dim'),
    Real(1e-4, 1e-1, prior='log-uniform', name='lr'),
    Integer(64, 512, name='batch_size')
]

res = gp_minimize(objective, search_space, n_calls=25, random_state=42)

print("\nBest Parameters Found:")
print(f"latent_dim: {res.x[0]}")
print(f"learning_rate: {res.x[1]}")
print(f"batch_size: {res.x[2]}")
