

import numpy as np
import pandas as pd
import os
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid
import matplotlib.pyplot as plt

def generate_structured_context(
    latent_path="data/context_latents.npy",
    output_path="data/structured_context.npy",
    levels=[5, 10, 15, 20],
    subset_size=10000,
    seed=42
):
    # Load latent context vectors
    latents = np.load(latent_path)
    n_samples = latents.shape[0]
    np.random.seed(seed)

    # Normalize
    scaler = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)

    # Subsample for clustering
    idx = np.random.choice(n_samples, size=min(subset_size, n_samples), replace=False)
    latents_subset = latents_scaled[idx]

    # Ward linkage on subset
    linkage_matrix = linkage(latents_subset, method='ward')

    # Optional dendrogram
    os.makedirs("visuals", exist_ok=True)
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    plt.title("Dendrogram (Subset of Latent Contexts)")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig("visuals/context_dendrogram.png")
    print("Dendrogram saved to visuals/context_dendrogram.png")

    structured_contexts = []

    for k in levels:
        print(f"Assigning {k} clusters...")

        # Cluster the subset
        subset_labels = fcluster(linkage_matrix, k, criterion='maxclust')

        # Fit centroid classifier
        clf = NearestCentroid()
        clf.fit(latents_subset, subset_labels)

        # Predict full dataset cluster labels
        full_labels = clf.predict(latents_scaled)
        structured_contexts.append(full_labels.reshape(-1, 1))

    # Stack cluster labels for each level into full structured context
    structured_matrix = np.hstack(structured_contexts)
    np.save(output_path, structured_matrix.astype(np.int32))
    print(f"Structured context saved to {output_path}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    generate_structured_context()
