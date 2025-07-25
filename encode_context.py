import torch
import numpy as np
import pandas as pd
import os

def encode_context(parquet_path="data/yelp_filtered.parquet", encoder_path="models/ae_encoder.pt", output_path="data/context_latents.npy"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_parquet(parquet_path)
    city_columns = [col for col in df.columns if col.startswith("city_")]
    context_columns = ["biz_stars", "review_count"] + city_columns

    context_matrix = df[context_columns].values.astype(np.float32)
    x = torch.tensor(context_matrix, dtype=torch.float32).to(device)

    encoder = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 3),
        torch.nn.ReLU()
    ).to(device)

    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()

    with torch.no_grad():
        encoded = encoder(x).cpu().numpy()

    np.save(output_path, encoded)
    print(f"Saved encoded context vectors to {output_path}")

if __name__ == "__main__":
    encode_context()
