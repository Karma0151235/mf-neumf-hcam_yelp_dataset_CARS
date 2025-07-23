import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import load_yelp, preprocess

def explore_yelp_dataset():
    # Load and preprocess data
    raw_df = load_yelp('datasets/', sample_size=500000)
    df, context_matrix = preprocess(raw_df, min_uc=3, min_ic=3)

    print("=== Basic Dataset Info ===")
    print(f"Total rows: {len(df)}")
    print(f"Unique users: {df['user'].nunique()}")
    print(f"Unique items: {df['item'].nunique()}")
    print(f"Rating range: {df['rating'].min()} to {df['rating'].max()}")
    print(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Number of latent context dimensions: {context_matrix.shape[1]}")

    print("\n=== Raw Data Columns ===")
    print(raw_df.columns.tolist())

    if 'attributes' in raw_df.columns:
        print("\nSample business attributes:")
        print(raw_df['attributes'].dropna().iloc[0])

    print("\n=== Rating Distribution ===")
    print(df['rating'].describe())
    df['rating'].hist(bins=20)
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    print("\n=== User Activity Distribution ===")
    user_counts = df['user'].value_counts()
    user_counts.hist(bins=50)
    plt.title("Ratings per User")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.show()

    print("\n=== Item Popularity Distribution ===")
    item_counts = df['item'].value_counts()
    item_counts.hist(bins=50)
    plt.title("Ratings per Item")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Number of Items")
    plt.tight_layout()
    plt.show()

    return df, context_matrix

if __name__ == "__main__":
    explore_yelp_dataset()
