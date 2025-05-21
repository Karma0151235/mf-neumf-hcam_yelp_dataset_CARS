import torch
from torch import nn

class ContextAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # turning inputs into latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, latent_dim)
        )
        # decoding latent back to input
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder
        return x_recon, z
