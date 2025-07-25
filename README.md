# Context-Aware Recommendation with HCAM-NeuMF

This repository contains the full implementation of a recommendation system benchmark study, based on the paper [*Context-Aware Recommendations Based on Deep Learning Frameworks*](https://doi.org/10.1145/3386243). The project compares three models—**Matrix Factorization (MF)**, **Neural Matrix Factorization (NeuMF)**, and a **Hierarchical Context-Aware NeuMF (HCAM-NeuMF)**—on the Yelp dataset using both random and time-based splits.

---

## Project Structure

```bash
.
├── data/
│   ├── yelp_filtered.parquet     # Preprocessed dataset (required)
│   ├── context_latents.npy       # Latent context vectors (autoencoder output)
│   ├── structured_context.npy    # Clustered context labels (HCAM input)
├── figures/                      # Auto-generated plots (loss curves, dendrograms, architecture)
├── journals/                      # Journals related to the project
├── models/                       # Model architectures for MF, NeuMF, HCAM-NeuMF
├── utils/                        # Utilities for preprocessing, evaluation, splitting, visualization
├── train_autoencoder.py         # Trains the autoencoder for context compression
├── encode_context.py            # Encodes context features into latent vectors
├── generate_structured_context.py # Clusters latent vectors into structured context
├── run_mf.py                     # Trains & evaluates Matrix Factorization model
├── run_neumf.py                  # Trains & evaluates NeuMF model
├── run_hcamneumf.py              # Trains & evaluates HCAM-NeuMF
├── optimize_mf.py               # Bayesian optimization for MF
├── optimize_neumf.py           # Bayesian optimization for NeuMF
├── optimize_hcamneumf.py       # Bayesian optimization for HCAM-NeuMF
└── README.md
```

## Setup 

1. **Clone the Repository**
   ```
   git clone https://github.com/Karma0151235/mf-neumf-hcam_yelp_dataset_CARS.git
   cd mf-neumf-hcam_yelp_dataset_CARS
   ```

2. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Data Preparation**
   
   <ol>Download the [Yelp dataset files](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) (business.json, review.json, etc.) into the datasets/ directory. </ol>

   <ol>Important: Ensure that all scripts reference the dataset path. </ol>
   
