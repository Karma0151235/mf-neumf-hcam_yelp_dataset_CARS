# --- evaluation.py ---

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from torch.utils.data import TensorDataset, DataLoader

def evaluate_model(model, test_df, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    users = torch.tensor(test_df['user'].values, dtype=torch.long)
    items = torch.tensor(test_df['item'].values, dtype=torch.long)
    ratings = torch.tensor(test_df['rating'].values, dtype=torch.float)

    loader = DataLoader(TensorDataset(users, items, ratings), batch_size=batch_size)

    preds, trues = [], []
    with torch.no_grad():
        for u, i, r in loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            pred = model(u, i)
            preds.append(pred.cpu().numpy())
            trues.append(r.cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    rmse = mean_squared_error(trues, preds, squared=False)
    mae = mean_absolute_error(trues, preds)
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return rmse, mae
