import matplotlib.pyplot as plt
import numpy as np

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
    import matplotlib.pyplot as plt

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

