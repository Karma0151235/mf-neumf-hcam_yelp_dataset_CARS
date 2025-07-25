
import torch
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from utils.autoencoder import Autoencoder
from sklearn.model_selection import train_test_split
import os

def train_autoencoder(context_matrix, latent_dim=3, epochs=20, batch_size=256, lr=0.001):
    input_dim = context_matrix.shape[1]
    model = Autoencoder(input_dim, latent_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x = torch.tensor(context_matrix, dtype=torch.float32).to(device)
    dataset = TensorDataset(x)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            _, output = model(batch)
            loss = loss_fn(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Save encoder only
    torch.save(model.encoder.state_dict(), "models/ae_encoder.pt")
    print("Saved encoder to models/ae_encoder.pt")

if __name__ == "__main__":
    df = pd.read_parquet("data/yelp_filtered.parquet")

    # Keep only columns that are explicitly numeric and related to context
    context_cols = [col for col in df.columns if col.startswith('biz_stars') or col.startswith('review_count') or col.startswith('city_')]
    ctx = df[context_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32').values  # <- clean & numeric

    train_autoencoder(ctx, latent_dim=3)

