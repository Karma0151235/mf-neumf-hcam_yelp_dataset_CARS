import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage


def plot_fold_losses(all_losses, title="Training Loss per Fold"):
    plt.figure(figsize=(10, 6))
    for idx, losses in enumerate(all_losses):
        plt.plot(losses, label=f"Fold {idx+1}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_train_vs_test(train_losses, val_losses, title="Train vs Validation Loss per Fold"):
    plt.figure(figsize=(10, 6))
    for idx, (train, val) in enumerate(zip(train_losses, val_losses)):
        plt.plot(train, label=f"Fold {idx+1} - Train")
        plt.plot(val, label=f"Fold {idx+1} - Val", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_model_comparison_losses(loss_lists, labels):
    plt.figure(figsize=(10, 6))
    for losses, label in zip(loss_lists, labels):
        avg_losses = np.mean(np.array(losses), axis=0)
        plt.plot(avg_losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Comparison Across Models")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_model_comparison_metrics(rmse_list, mae_list, labels):
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, rmse_list, width, label='RMSE')
    plt.bar(x + width/2, mae_list, width, label='MAE')
    plt.ylabel('Error')
    plt.title('RMSE and MAE Comparison')
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_train_vs_val_mae(train_maes, val_maes, title="Train vs Validation MAE per Fold"):
    plt.figure(figsize=(10, 6))
    for idx, (train, val) in enumerate(zip(train_maes, val_maes)):
        plt.plot(train, label=f"Fold {idx+1} - Train MAE")
        plt.plot(val, label=f"Fold {idx+1} - Val MAE", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_val_vs_test_metrics(val_rmses, test_rmses, val_maes, test_maes, labels):
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, val_rmses, width, label='Validation RMSE')
    plt.bar(x + width/2, test_rmses, width, label='Test RMSE')
    plt.xticks(x, labels)
    plt.ylabel("RMSE")
    plt.title("Validation vs Test RMSE")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, val_maes, width, label='Validation MAE')
    plt.bar(x + width/2, test_maes, width, label='Test MAE')
    plt.xticks(x, labels)
    plt.ylabel("MAE")
    plt.title("Validation vs Test MAE")
    plt.legend()
    plt.tight_layout()
    plt.show()
