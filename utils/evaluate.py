import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch

def evaluate(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in data_loader:
            u, i, r = [x.to(device) for x in batch]
            pred = model(u, i)
            preds.append(pred.cpu().numpy())
            trues.append(r.cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    rmse = mean_squared_error(trues, preds, squared=True) ** 0.5
    mae = mean_absolute_error(trues, preds)
    print(f"Evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}")
    return rmse, mae