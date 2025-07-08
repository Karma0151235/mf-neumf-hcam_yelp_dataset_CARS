import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from utils.data_loader import load_yelp, preprocess
from utils.splitter import random_split_3way, split_time
from utils.evaluate import evaluate_model
from utils.visualize import (
    plot_fold_losses, plot_train_vs_test,
    plot_model_comparison_losses, plot_model_comparison_metrics,
    plot_latent_space_3d, plot_ahc_dendrogram,
    plot_val_vs_test_metrics
)
from utils.autoencoder import Autoencoder
from models.hcamneumf import HCAMNeuMF
from run_mf import run_mf
from run_neumf import run_neumf
from run_hcamneumf import run_hcamneumf

if __name__ == '__main__':
    df = pd.read_parquet("data/yelp_filtered.parquet")
    context_matrix = np.load("data/context_latents.npy")

    print("\n[1] Running MF Model (10-fold)...")
    mf_results = run_mf(df)

    print("\n[2] Running NeuMF Model (10-fold)...")
    neumf_results = run_neumf(df)

    print("\n[3] Running HCAM-NeuMF Model (10-fold)...")
    hcam_results = run_hcamneumf(df, context_matrix)

    # === Train vs Validation Loss: 10-Fold ===
    plot_train_vs_test(mf_results['train_losses'], mf_results['val_losses'], "MF Train vs Val Loss (10-fold)")
    plot_train_vs_test(neumf_results['train_losses'], neumf_results['val_losses'], "NeuMF Train vs Val Loss (10-fold)")
    plot_train_vs_test(hcam_results['train_losses'], hcam_results['val_losses'], "HCAM-NeuMF Train vs Val Loss (10-fold)")

    # === Validation Loss Comparison (10-Fold) ===
    plot_model_comparison_losses([
        mf_results['val_losses'],
        neumf_results['val_losses'],
        hcam_results['val_losses']
    ], labels=['MF', 'NeuMF', 'HCAM-NeuMF'], title="Val Loss Comparison (10-fold)")

    # === RMSE / MAE Comparison (10-Fold) ===
    plot_model_comparison_metrics([
        mf_results['rmse'], neumf_results['rmse'], hcam_results['rmse']
    ], [
        mf_results['mae'], neumf_results['mae'], hcam_results['mae']
    ], labels=['MF', 'NeuMF', 'HCAM-NeuMF'], title="Test RMSE & MAE (10-fold)")

    # === Time-Based Evaluation ===
    print("\n[4] Time-Based Evaluation")
    mf_tb_results = run_mf(df, time_based=True)
    neumf_tb_results = run_neumf(df, time_based=True)
    hcam_tb_results = run_hcamneumf(df, context_matrix, time_based=True)

    # === Train vs Validation Loss: Time-Based ===
    plot_train_vs_test(mf_tb_results['train_losses'], mf_tb_results['val_losses'], "MF Train vs Val Loss (Time-Based)")
    plot_train_vs_test(neumf_tb_results['train_losses'], neumf_tb_results['val_losses'], "NeuMF Train vs Val Loss (Time-Based)")
    plot_train_vs_test(hcam_tb_results['train_losses'], hcam_tb_results['val_losses'], "HCAM-NeuMF Train vs Val Loss (Time-Based)")

    # === RMSE / MAE Comparison (Time-Based) ===
    plot_model_comparison_metrics([
        mf_tb_results['rmse'], neumf_tb_results['rmse'], hcam_tb_results['rmse']
    ], [
        mf_tb_results['mae'], neumf_tb_results['mae'], hcam_tb_results['mae']
    ], labels=['MF (TB)', 'NeuMF (TB)', 'HCAM-NeuMF (TB)'], title="Test RMSE & MAE (Time-Based)")

    # === Context Visualizations ===
    plot_latent_space_3d(np.load("data/context_latents.npy"))
    try:
        plot_ahc_dendrogram(np.load("data/context_latents.npy"))
    except Exception as e:
        print("AHC Dendrogram failed:", str(e))

    print("\n✅ Final comparison complete.")