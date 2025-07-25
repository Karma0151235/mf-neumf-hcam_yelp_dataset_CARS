import pandas as pd
from sklearn.model_selection import KFold, train_test_split


def split_kfold(df, k=10, seed=42):
    """
    Yields train/val splits using KFold Cross Validation.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle once

    for train_idx, val_idx in kf.split(df):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        yield train_df, val_df


def split_time(df, ratio=0.8):
    """
    Splits the data based on timestamp.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * ratio)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, val_df


def random_split_3way(df, seed):
    """
    Splits the data randomly into 80% train, 10% val, 10% test (used in MF replication).
    """
    train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=seed)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1111, random_state=seed)  # 0.1111 ≈ 10% of original
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
