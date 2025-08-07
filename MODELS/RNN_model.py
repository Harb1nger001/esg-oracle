import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ---------------------- PATHS ----------------------

model_save_path = r"D:\Quantum ESG forcasting\results\models\rnn_model.pth"
prediction_save_path = r"D:\Quantum ESG forcasting\results\predictions\rnn_predictions.csv"
metrics_save_path = r"D:\Quantum ESG forcasting\results\metrics\rnn_metrics.csv"

# === Load the combined dataset ===
file_path = r"D:\Quantum ESG forcasting\results\esg_scores_grouped.csv"
df = pd.read_csv(file_path)

# === Define X and Y columns ===
x_columns = [
    "Carbon dioxide (CO2) emissions (total) excluding LULUCF (% change from 1990)",
    "Carbon intensity of GDP (kg CO2e per 2021 PPP $ of GDP)",
    "Current health expenditure per capita (current US$)",
    "Employment to population ratio, 15+, total (%) (modeled ILO estimate)",
    "Forest area (% of land area)",
    "Literacy rate, youth total (% of people ages 15-24)",
    "PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)",
    "Political Stability and Absence of Violence/Terrorism: Estimate",
    "Renewable energy consumption (% of total final energy consumption)",
    "wgi_pivoted_cleaned"
]

y_columns = ["E_Score", "S_Score", "G_Score", "ESG_Score"]

# === Encode Country Name ===
le = LabelEncoder()
df['Country Name'] = le.fit_transform(df['Country Name'])

# === Create full multi-index from all countries and years ===
all_countries = df['Country Name'].unique()
all_years = np.arange(1991, 2025)

full_index = pd.MultiIndex.from_product([all_countries, all_years], names=['Country Name', 'Year'])

# === Reindex to ensure all (country, year) pairs are present ===
df.set_index(['Country Name', 'Year'], inplace=True)
df = df.reindex(full_index)

# === Fill missing values with 0 ===
df[x_columns + y_columns] = df[x_columns + y_columns].fillna(0)

# === Reshape to 3D tensors ===
num_countries = len(all_countries)
num_years = len(all_years)
num_x_feats = len(x_columns)
num_y_feats = len(y_columns)

X_tensor = torch.tensor(df[x_columns].values, dtype=torch.float32).view(num_countries, num_years, num_x_feats)
Y_tensor = torch.tensor(df[y_columns].values, dtype=torch.float32).view(num_countries, num_years, num_y_feats)

# === Optional debug prints ===
print(f"X_tensor.shape: {X_tensor.shape}")
print(f"Y_tensor.shape: {Y_tensor.shape}")
print(f"Any NaNs in X? {torch.isnan(X_tensor).any().item()}")
print(f"Any NaNs in Y? {torch.isnan(Y_tensor).any().item()}")


# ------------------ SPLIT & LOAD ------------------
X_train, X_test, Y_train, Y_test = train_test_split(X_tensor, Y_tensor, test_size=0.2, random_state=42)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=32)

# ------------------ RNN MODEL ------------------
class ESG_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(ESG_RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

model = ESG_RNN(input_size=X_tensor.shape[2], hidden_size=128, output_size=4)
criterion = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ------------------ TRAINING ------------------
print("🚀 Training RNN Model...")
for epoch in range(200):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/200")
    for batch_x, batch_y in loop:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y[:, -1, :])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(train_loader):.4f}")

# ------------------ EVALUATION ------------------
model.eval()
with torch.no_grad():
    y_pred = model(X_test).numpy()
    y_true = Y_test[:, -1, :].numpy()

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n📊 Evaluation Metrics:")
print(f"  MSE: {mse:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  R2 Score: {r2:.4f}")

# ------------------ SAVE OUTPUTS ------------------
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
pd.DataFrame(y_pred, columns=['E_Score', 'S_Score', 'G_Score', 'ESG_Score']).to_csv(prediction_save_path, index=False)
pd.DataFrame({
    'Metric': ['MSE', 'MAE', 'R2'],
    'Value': [mse, mae, r2]
}).to_csv(metrics_save_path, index=False)

print("\n✅ All done. Model, predictions, and metrics saved successfully! 🍻")

# ------------------ PREDICT 2025 & 2026 ------------------

model.eval()
future_years = [2025, 2026]
all_preds = []
all_countries = []

with torch.no_grad():
    # Start from the last available time slice (e.g., ending at 2024)
    last_input = X_tensor.clone()  # Shape: [num_countries, time_steps, features]
    current_input = last_input

    for year in future_years:
        pred = model(current_input)  # Shape: [num_countries, 4]
        all_preds.append(pred.numpy())

        # Create new input by appending predicted values (assumes ESG scores go at the end of each feature vector)
        # So we update only the last 4 features with the predicted ESG values for each country
        next_input = current_input[:, 1:, :].clone()  # Drop first timestep
        next_input = torch.cat([next_input, current_input[:, -1:, :].clone()], dim=1)  # Repeat last step
        next_input[:, -1, -4:] = pred  # Replace ESG scores
        current_input = next_input

# ------------------ SAVE FUTURE PREDICTIONS ------------------

future_df = pd.DataFrame()
for i, year in enumerate(future_years):
    year_preds = all_preds[i]
    temp_df = pd.DataFrame(year_preds, columns=["E_Score", "S_Score", "G_Score", "ESG_Score"])
    temp_df["Year"] = year
    temp_df["Country"] = le.inverse_transform(np.arange(281))  # 281 = total number of countries
    future_df = pd.concat([future_df, temp_df], ignore_index=True)

# Save to CSV
output_dir = r"D:\Quantum ESG forcasting\results\predictions"
os.makedirs(output_dir, exist_ok=True)

future_df.to_csv(os.path.join(output_dir, "esg_forecasts_2025_2026.csv"), index=False)
print("📈 Future ESG forecasts for 2025 and 2026 saved.")
