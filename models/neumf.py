# --- models/neumf.py ---

import torch
from torch import nn


class GMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim=32, init_std=0.01):
        super(GMF, self).__init__()
        self.embedding_user = nn.Embedding(num_users, latent_dim)
        self.embedding_item = nn.Embedding(num_items, latent_dim)
        self.affine_output = nn.Linear(latent_dim, 1)
        self.logistic = nn.Sigmoid()

        self._init_weights(init_std)

    def forward(self, user_indices, item_indices):
        u = self.embedding_user(user_indices)
        i = self.embedding_item(item_indices)
        x = torch.mul(u, i)
        x = self.affine_output(x)
        return self.logistic(x)

    def _init_weights(self, std):
        nn.init.normal_(self.embedding_user.weight, std=std)
        nn.init.normal_(self.embedding_item.weight, std=std)
        nn.init.normal_(self.affine_output.weight, std=std)
        self.affine_output.bias.data.zero_()


class MLP(nn.Module):
    def __init__(self, num_users, num_items, layers=[64, 32, 16, 8], init_std=0.01):
        super(MLP, self).__init__()
        assert layers[0] % 2 == 0, "First layer size should be divisible by 2."

        self.embedding_user = nn.Embedding(num_users, layers[0] // 2)
        self.embedding_item = nn.Embedding(num_items, layers[0] // 2)

        self.fc_layers = nn.ModuleList()
        for in_size, out_size in zip(layers[:-1], layers[1:]):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        self.affine_output = nn.Linear(layers[-1], 1)
        self.logistic = nn.Sigmoid()

        self._init_weights(init_std)

    def forward(self, user_indices, item_indices):
        u = self.embedding_user(user_indices)
        i = self.embedding_item(item_indices)
        x = torch.cat([u, i], dim=-1)
        for layer in self.fc_layers:
            x = torch.relu(layer(x))
        x = self.affine_output(x)
        return self.logistic(x)

    def _init_weights(self, std):
        nn.init.normal_(self.embedding_user.weight, std=std)
        nn.init.normal_(self.embedding_item.weight, std=std)
        for fc in self.fc_layers:
            nn.init.normal_(fc.weight, std=std)
            fc.bias.data.zero_()
        nn.init.normal_(self.affine_output.weight, std=std)
        self.affine_output.bias.data.zero_()


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim_mf=16, latent_dim_mlp=32, mlp_layers=[64, 32, 16, 8], init_std=0.01):
        super(NeuMF, self).__init__()
        self.embedding_user_mlp = nn.Embedding(num_users, latent_dim_mlp)
        self.embedding_item_mlp = nn.Embedding(num_items, latent_dim_mlp)
        self.embedding_user_mf = nn.Embedding(num_users, latent_dim_mf)
        self.embedding_item_mf = nn.Embedding(num_items, latent_dim_mf)

        self.fc_layers = nn.ModuleList()
        for in_size, out_size in zip(mlp_layers[:-1], mlp_layers[1:]):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        predict_size = mlp_layers[-1] + latent_dim_mf
        self.affine_output = nn.Linear(predict_size, 1)
        self.logistic = nn.Sigmoid()

        self._init_weights(init_std)

    def forward(self, user_indices, item_indices):
        u_mlp = self.embedding_user_mlp(user_indices)
        i_mlp = self.embedding_item_mlp(item_indices)
        u_mf = self.embedding_user_mf(user_indices)
        i_mf = self.embedding_item_mf(item_indices)

        x_mlp = torch.cat([u_mlp, i_mlp], dim=-1)
        for layer in self.fc_layers:
            x_mlp = torch.relu(layer(x_mlp))

        x_mf = torch.mul(u_mf, i_mf)
        x = torch.cat([x_mlp, x_mf], dim=-1)
        x = self.affine_output(x)
        return self.logistic(x)

    def _init_weights(self, std):
        for emb in [self.embedding_user_mlp, self.embedding_item_mlp, self.embedding_user_mf, self.embedding_item_mf]:
            nn.init.normal_(emb.weight, std=std)
        for fc in self.fc_layers:
            nn.init.normal_(fc.weight, std=std)
            fc.bias.data.zero_()
        nn.init.normal_(self.affine_output.weight, std=std)
        self.affine_output.bias.data.zero_()
