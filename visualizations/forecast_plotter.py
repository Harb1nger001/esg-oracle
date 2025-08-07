import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# File paths
real_data_path = r"D:\Quantum ESG forcasting\results\esg_scores_grouped.csv"
rnn_pred_path = r"D:\Quantum ESG forcasting\results\predictions\esg_forecasts_2025_2026.csv"
lstm_pred_path = r"D:\Quantum ESG forcasting\results\predictions\esg_forecasts_lstm_2025_2026.csv"
output_dir = r"D:\Quantum ESG forcasting\results\visualizations"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load datasets
real_df = pd.read_csv(real_data_path)
rnn_df = pd.read_csv(rnn_pred_path)
lstm_df = pd.read_csv(lstm_pred_path)

# Score columns to visualize
score_columns = ["E_Score", "S_Score", "G_Score", "ESG_Score"]

# Plotting
for score in score_columns:
    plt.figure(figsize=(10, 6))

    sns.kdeplot(real_df[score].dropna(), label='Real', linewidth=2)
    sns.kdeplot(rnn_df[score].dropna(), label='RNN', linestyle='--', linewidth=2)
    sns.kdeplot(lstm_df[score].dropna(), label='LSTM', linestyle=':', linewidth=2)

    plt.title(f'Distribution of {score}: Real vs RNN vs LSTM')
    plt.xlabel(score)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

    # Save the plot
    filename = f"{score}_distribution_real_rnn_lstm.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
