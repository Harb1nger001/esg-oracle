import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import time  # Just for artificial delay to show progress nicely

# File paths
real_data_path = r"D:\Quantum ESG forcasting\results\esg_scores_grouped.csv"
synthetic_data_path = r"D:\Quantum ESG forcasting\data\synthetic\cvae_synthetic_data.csv"
output_dir = r"D:\Quantum ESG forcasting\results\visualizations"

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load datasets
real_df = pd.read_csv(real_data_path)
synthetic_df = pd.read_csv(synthetic_data_path)

# Prepare real ESG scores
real_scores = real_df[["E_Score", "S_Score", "G_Score", "ESG_Score"]].dropna().copy()
real_scores["Source"] = "Real"

# Prepare synthetic ESG scores
val_array = synthetic_df["Value"].values
num_rows = val_array.shape[0] // 4
reshaped_vals = val_array[:num_rows * 4].reshape(num_rows, 4)

synthetic_scores = pd.DataFrame(reshaped_vals, columns=["E_Score", "S_Score", "G_Score", "ESG_Score"])
synthetic_scores["Source"] = "Synthetic"

# Combine datasets
combined_df = pd.concat([real_scores, synthetic_scores], ignore_index=True)

# Standardize the data
features = ["E_Score", "S_Score", "G_Score", "ESG_Score"]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(combined_df[features])

# Show progress bar for t-SNE + plotting (2 steps total)
with tqdm(total=2, desc="Generating t-SNE plot", ncols=80) as pbar:
    # Step 1: t-SNE projection
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(scaled_features)
    pbar.update(1)

    # Add results
    combined_df["TSNE1"] = tsne_results[:, 0]
    combined_df["TSNE2"] = tsne_results[:, 1]

    # Step 2: Plot and save
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=combined_df, x="TSNE1", y="TSNE2", hue="Source", alpha=0.6, s=60, palette="deep")
    plt.title("t-SNE Projection: Real vs Synthetic ESG Scores")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "tsne_real_vs_synthetic.png"))
    plt.close()
    pbar.update(1)
