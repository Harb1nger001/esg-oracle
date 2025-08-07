import pandas as pd
import matplotlib.pyplot as plt
import os

# File paths
real_data_path = r"D:\Quantum ESG forcasting\results\esg_scores_grouped.csv"
rnn_pred_path = r"D:\Quantum ESG forcasting\results\predictions\esg_forecasts_2025_2026.csv"
lstm_pred_path = r"D:\Quantum ESG forcasting\results\predictions\esg_forecasts_lstm_2025_2026.csv"
output_path = r"D:\Quantum ESG forcasting\results\visualizations"

# Load real ESG scores (up to 2024)
real_df = pd.read_csv(real_data_path)
real_df = real_df[["Year", "ESG_Score"]]
real_df_grouped = real_df.groupby("Year").mean().reset_index()

# Load RNN predictions
rnn_df = pd.read_csv(rnn_pred_path)
rnn_df_grouped = rnn_df.groupby("Year")["ESG_Score"].mean().reset_index()

# Load LSTM predictions
lstm_df = pd.read_csv(lstm_pred_path)
lstm_df_grouped = lstm_df.groupby("Year")["ESG_Score"].mean().reset_index()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(real_df_grouped["Year"], real_df_grouped["ESG_Score"], label="Real", marker='o')
plt.plot(rnn_df_grouped["Year"], rnn_df_grouped["ESG_Score"], label="RNN Predicted (2025–26)", marker='^')
plt.plot(lstm_df_grouped["Year"], lstm_df_grouped["ESG_Score"], label="LSTM Predicted (2025–26)", marker='s')

plt.title("ESG Score Trend: Real vs RNN vs LSTM Predictions")
plt.xlabel("Year")
plt.ylabel("ESG Score")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save output
os.makedirs(output_path, exist_ok=True)
plt.savefig(os.path.join(output_path, "esg_trend_comparison_future.png"))
plt.show()
