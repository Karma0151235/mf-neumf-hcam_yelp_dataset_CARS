import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from models.mf import MF
from models.neumf import NeuMF
from models.hcamneumf import HCAMNeuMF

from torchviz import make_dot


def visualize_model_architecture(model, example_inputs, title="Model"):
    model.eval()
    output = model(*example_inputs)
    dot = make_dot(output, params=dict(list(model.named_parameters())))
    dot.format = 'png'
    dot.directory = 'figures'
    dot.render(title)
    print(f"Saved {title} architecture to figures/{title}.png")


def plot_latent_space_3d(context_latents):
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(context_latents)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], s=5, alpha=0.7)
    ax.set_title("Context Latent Space (3D PCA)")
    plt.tight_layout()
    plt.show()


def plot_ahc_dendrogram(context_latents, sample_size=10000):
    if context_latents.shape[0] > sample_size:
        idx = np.random.choice(context_latents.shape[0], sample_size, replace=False)
        context_latents = context_latents[idx]

    linkage_matrix = linkage(context_latents, method='ward')
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, truncate_mode='lastp', p=30, leaf_rotation=90.,
               leaf_font_size=10., show_contracted=True)
    plt.title("AHC Dendrogram (Ward’s Method)")
    plt.xlabel("Context Points")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()


def plot_latent_distribution(context_latents):
    plt.figure(figsize=(10, 6))
    for i in range(context_latents.shape[1]):
        plt.hist(context_latents[:, i], bins=50, alpha=0.5, label=f"Dim {i}")
    plt.legend()
    plt.title("Context Latent Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def visualize_model_architecture_svg(model, example_inputs, title="Model"):
    model.eval()
    output = model(*example_inputs)
    dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=False, show_saved=False)
    dot.format = 'svg'
    dot.directory = 'figures'
    dot.render(title)
    print(f"Saved {title} architecture to figures/{title}.svg")

if __name__ == "__main__":
    # Load context latents
    context_latents = np.load("data/context_latents.npy")
    print("Latent shape:", context_latents.shape)

    # Plot latent space visuals
    plot_latent_distribution(context_latents)
    plot_latent_space_3d(context_latents)
    plot_ahc_dendrogram(context_latents)

    # Visualize model architectures
    num_users, num_items, context_dim = 100, 200, context_latents.shape[1]
    user = torch.tensor([0])
    item = torch.tensor([0])
    context = torch.tensor(context_latents[0:1], dtype=torch.float32)

    print("Rendering model architectures to /figures...")

    mf_model = MF(num_users, num_items, latent_dim=8)
    visualize_model_architecture(mf_model, (user, item), title="MF_Architecture")

    neumf_model = NeuMF(num_users, num_items, latent_dim_mf=16, latent_dim_mlp=16, mlp_layers=[32, 16, 8], dropout=0.31778351694762974)
    visualize_model_architecture(neumf_model, (user, item), title="NeuMF_Architecture")

    hcam_model = HCAMNeuMF(num_users, num_items, context_dim=context_dim, latent_dim_mf=16, latent_dim_mlp=16, mlp_layers=[32, 16, 8], dropout=0.2)
    visualize_model_architecture(hcam_model, (user, item, context), title="HCAMNeuMF_Architecture")

    visualize_model_architecture_svg(mf_model, (user, item), title="MF_ArchitectureSVG")
    visualize_model_architecture_svg(neumf_model, (user, item), title="NeuMF_ArchitectureSVG")
    visualize_model_architecture_svg(hcam_model, (user, item, context), title="HCAMNeuMF_ArchitectureSVG")



