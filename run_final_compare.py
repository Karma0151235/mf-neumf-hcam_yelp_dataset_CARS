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
    plot_val_vs_test_metrics, plot_train_vs_val_mae  # Include new function
)
from utils.autoencoder import Autoencoder
from models.hcamneumf import HCAMNeuMF
from run_mf import run_mf
from run_neumf import run_neumf
from run_hcamneumf import run_hcamneumf

if __name__ == '__main__':
    path_json_dir = 'datasets/'
    raw_df = load_yelp(path_json_dir, sample_size=500000)
    df, _ = preprocess(raw_df, min_uc=3, min_ic=3)
    context_matrix = np.load("data/context_latents.npy")

    # === Time-Based Evaluation ===
    print("\n[1] Time-Based Evaluation")
    mf_tb_results = run_mf(df, time_based=True)
    neumf_tb_results = run_neumf(df, time_based=True)
    hcam_tb_results = run_hcamneumf(df, context_matrix, time_based=True)

    # === Train vs Validation Loss (Time-Based) ===
    plot_train_vs_test(mf_tb_results['train_losses'], mf_tb_results['val_losses'], "MF Train vs Val Loss (Time-Based)")
    plot_train_vs_test(neumf_tb_results['train_losses'], neumf_tb_results['val_losses'], "NeuMF Train vs Val Loss (Time-Based)")
    plot_train_vs_test(hcam_tb_results['train_losses'], hcam_tb_results['val_losses'], "HCAM-NeuMF Train vs Val Loss (Time-Based)")

    # === Train vs Validation MAE (Time-Based) ===
    if 'train_maes' in mf_tb_results and 'val_maes' in mf_tb_results:
        plot_train_vs_val_mae(mf_tb_results['train_maes'], mf_tb_results['val_maes'], "MF Train vs Val MAE (Time-Based)")
    if 'train_maes' in neumf_tb_results and 'val_maes' in neumf_tb_results:
        plot_train_vs_val_mae(neumf_tb_results['train_maes'], neumf_tb_results['val_maes'], "NeuMF Train vs Val MAE (Time-Based)")
    if 'train_maes' in hcam_tb_results and 'val_maes' in hcam_tb_results:
        plot_train_vs_val_mae(hcam_tb_results['train_maes'], hcam_tb_results['val_maes'], "HCAM-NeuMF Train vs Val MAE (Time-Based)")

    # === RMSE / MAE Comparison (Time-Based) ===
    plot_model_comparison_metrics([
        mf_tb_results['rmse'], neumf_tb_results['rmse'], hcam_tb_results['rmse']
    ], [
        mf_tb_results['mae'], neumf_tb_results['mae'], hcam_tb_results['mae']
    ], labels=['MF (TB)', 'NeuMF (TB)', 'HCAM-NeuMF (TB)'])

    # === 10-Fold Evaluation ===
    print("\n[2] Running MF Model (10-fold)...")
    mf_results = run_mf(df)

    print("\n[3] Running NeuMF Model (10-fold)...")
    neumf_results = run_neumf(df)

    print("\n[4] Running HCAM-NeuMF Model (10-fold)...")
    hcam_results = run_hcamneumf(df, context_matrix)

    # === Train vs Validation Loss (10-Fold) ===
    plot_train_vs_test(mf_results['train_losses'], mf_results['val_losses'], "MF Train vs Val Loss (10-fold)")
    plot_train_vs_test(neumf_results['train_losses'], neumf_results['val_losses'], "NeuMF Train vs Val Loss (10-fold)")
    plot_train_vs_test(hcam_results['train_losses'], hcam_results['val_losses'], "HCAM-NeuMF Train vs Val Loss (10-fold)")

    # === Train vs Validation MAE (10-Fold) ===
    if 'train_maes' in mf_results and 'val_maes' in mf_results:
        plot_train_vs_val_mae(mf_results['train_maes'], mf_results['val_maes'], "MF Train vs Val MAE (10-fold)")
    if 'train_maes' in neumf_results and 'val_maes' in neumf_results:
        plot_train_vs_val_mae(neumf_results['train_maes'], neumf_results['val_maes'], "NeuMF Train vs Val MAE (10-fold)")
    if 'train_maes' in hcam_results and 'val_maes' in hcam_results:
        plot_train_vs_val_mae(hcam_results['train_maes'], hcam_results['val_maes'], "HCAM-NeuMF Train vs Val MAE (10-fold)")

    # === RMSE / MAE Comparison (10-Fold) ===
    plot_model_comparison_metrics([
        mf_results['rmse'], neumf_results['rmse'], hcam_results['rmse']
    ], [
        mf_results['mae'], neumf_results['mae'], hcam_results['mae']
    ], labels=['MF', 'NeuMF', 'HCAM-NeuMF'])

    print("\nFinal comparison complete.")
