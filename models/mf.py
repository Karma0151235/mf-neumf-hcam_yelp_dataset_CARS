import torch.nn as nn

class MF(nn.Module):
    def __init__(self, n_users, n_items, latent_dim):
        super().__init__()
        self.u_emb = nn.Embedding(n_users, latent_dim)
        self.i_emb = nn.Embedding(n_items, latent_dim)

    def forward(self, u, i):
        # Dot Product
        return (self.u_emb(u) * self.i_emb(i)).sum(dim = 1)
