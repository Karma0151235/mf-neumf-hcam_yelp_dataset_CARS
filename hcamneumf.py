import torch
from torch import nn

class HCAMNeuMF(nn.Module):
    def __init__(self, num_users, num_items, context_dim,
                 latent_dim_mf=16, latent_dim_mlp=32, mlp_layers=[64, 32, 16, 8]):
        super(HCAMNeuMF, self).__init__()
        self.latent_dim_mf = latent_dim_mf
        self.latent_dim_mlp = latent_dim_mlp

        self.user_embedding_mf = nn.Embedding(num_users, latent_dim_mf)
        self.item_embedding_mf = nn.Embedding(num_items, latent_dim_mf)

        self.user_embedding_mlp = nn.Embedding(num_users, latent_dim_mlp)
        self.item_embedding_mlp = nn.Embedding(num_items, latent_dim_mlp)

        # First MLP layer expects user + item + context
        mlp_input_dim = latent_dim_mlp * 2 + context_dim
        mlp_layers_full = [mlp_input_dim] + mlp_layers

        self.mlp_layers = nn.ModuleList()
        for in_dim, out_dim in zip(mlp_layers_full[:-1], mlp_layers_full[1:]):
            self.mlp_layers.append(nn.Linear(in_dim, out_dim))
            self.mlp_layers.append(nn.ReLU())

        # Output layer expects concatenated GMF + MLP outputs
        self.output_layer = nn.Linear(latent_dim_mf + mlp_layers[-1], 1)

    def forward(self, user_ids, item_ids, context):
        # MF part
        user_mf = self.user_embedding_mf(user_ids)
        item_mf = self.item_embedding_mf(item_ids)
        mf_vector = user_mf * item_mf  # element-wise product

        # MLP part
        user_mlp = self.user_embedding_mlp(user_ids)
        item_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([user_mlp, item_mlp, context], dim=-1)

        x = mlp_input
        for layer in self.mlp_layers:
            x = layer(x)

        # Concatenate GMF and MLP
        final_vector = torch.cat([mf_vector, x], dim=-1)
        prediction = self.output_layer(final_vector)
        return prediction.squeeze()
