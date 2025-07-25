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
├── run_final_compare.py          # Centralized script, use for final run
├── optimize_mf.py               # Bayesian optimization for MF
├── optimize_neumf.py           # Bayesian optimization for NeuMF
├── optimize_hcamneumf.py       # Bayesian optimization for HCAM-NeuMF
└── README.md
```
<br>

## Setup 

### 1. Clone the Repository
   ```bash
   git clone https://github.com/Karma0151235/mf-neumf-hcam_yelp_dataset_CARS.git
   cd mf-neumf-hcam_yelp_dataset_CARS
   ```
<br>

### 2. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```
<br>

### 3. Data Preparation
   a. Download the [Yelp dataset files](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) (business.json, review.json, etc.) into the datasets/ directory. <br>
   
   b. Important: Ensure that _all_ scripts reference the dataset path.
<br>
<br>

### 4. Generate Context Vectors
   ```bash
   python train_autoencoder.py
   python encode_context.py
   python generate_structured_context.py
   ```
<br>

### 5. Train and Evaluate Models
   ```bash
   python run_final_compare.py
   ```
<br>

## Model Summary

| Model          | Description                                                                                                        |
| -------------- | ------------------------------------------------------------------------------------------------------------------ |
| **MF**         | Standard matrix factorization using element-wise product of user/item embeddings.                                  |
| **NeuMF**      | Combines MF and deep MLP components with learnable fusion.                                                         |
| **HCAM-NeuMF** | Extends NeuMF by incorporating structured context vectors derived from hierarchical clustering of latent features. |
<br>

## Evaluation Metrics

- RMSE (Root Mean Squared Error)
   
- MAE (Mean Absolute Error)
   
- Evaluation is done using:
   
   - 10-fold Cross-Validation

   - Time-based splitting (chronological 80/20 split)
<br>

## Hyperparameter Tuning

Bayesian Optimization using `scikit-optimize` was used to fine-tune hyperparameters (latent dimensions, learning rate, dropout). <br>
Due to time and compute constraints, not all hyperparameters (e.g. number of layers, batch size) were included in the search space. Some models were also manually tuned for practical performance.
<br>
<br>
<br>

## Visuals
The figures/ folder contains:

- Context latent distribution histogram
- Context latent space (PCA 3D scatter)
- AHC dendrogram of context clusters
- Train vs Validation loss/MAE curves
- Model architecture diagrams (torchviz)
- Yelp subset EDA visuals
<br>


