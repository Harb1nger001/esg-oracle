import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# -------------------------
# Configuration
# -------------------------
LATENT_DIM = 8
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001
INPUT_FILE = r"D:\\Quantum ESG forcasting\\results\\metrics\\zscore_normalized.csv"
OUTPUT_DATA_PATH = r"D:\\Quantum ESG forcasting\\data\\synthetic"
OUTPUT_MODEL_PATH = r"D:\\Quantum ESG forcasting\\results\\models"

# -------------------------
# Load and Normalize Condition (Year)
# -------------------------
df = pd.read_csv(INPUT_FILE)
all_numeric = df.select_dtypes(include='number').columns.tolist()
condition_col = 'Year'
feature_cols = [col for col in all_numeric if col != condition_col]

year_min = df[condition_col].min()
year_max = df[condition_col].max()
df[condition_col] = (df[condition_col] - year_min) / (year_max - year_min)

x_data = torch.tensor(df[feature_cols].values, dtype=torch.float32)
c_data = torch.tensor(df[[condition_col]].values, dtype=torch.float32)
dataset = TensorDataset(x_data, c_data)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------
# CVAE Model Definition
# -------------------------
class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)  # Prevent overflow
        std = torch.exp(0.5 * logvar) + 1e-8           # Add epsilon for stability
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, x, c):
        enc_input = torch.cat([x, c], dim=-1)
        h = self.encoder(enc_input)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        dec_input = torch.cat([z, c], dim=-1)
        recon_x = self.decoder(dec_input)
        return recon_x, mu, logvar

# -------------------------
# Loss Function (Huber + KL)
# -------------------------
def loss_function(recon_x, x, mu, logvar, beta=1e-4, delta=0.95):
    huber = nn.HuberLoss(delta=delta, reduction='mean')
    recon_loss = huber(recon_x, x)
    logvar = torch.clamp(logvar, min=-10, max=10)  # Optional: clamp here too
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div



# -------------------------
# Train Model
# -------------------------
model = CVAE(input_dim=x_data.shape[1], condition_dim=1, latent_dim=LATENT_DIM)
optimizer = torch.optim.Adagrad(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

print("\n🚀 Starting CVAE Training...")
for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    for x, c in tqdm(data_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        recon_x, mu, logvar = model(x, c)
        loss = loss_function(recon_x, x, mu, logvar, beta=0.1, delta=0.95)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch} | Loss: {total_loss:.4f}")

# -------------------------
# Generate Synthetic Data for Next 15 Years
# -------------------------
os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)

with torch.no_grad():
    future_years = torch.arange(year_max + 1, year_max + 16).float().unsqueeze(1)
    future_years = (future_years - year_min) / (year_max - year_min)

    samples_per_year = len(x_data) // len(torch.unique(c_data))

    generated_all = []
    for year in future_years:
        c = year.repeat(samples_per_year, 1)
        z = torch.randn(samples_per_year, LATENT_DIM)
        dec_input = torch.cat([z, c], dim=-1)
        generated = model.decoder(dec_input)
        combined = torch.cat([c, generated], dim=1)
        generated_all.append(combined)

    synthetic = torch.vstack(generated_all)
    synthetic[:, 0] = synthetic[:, 0] * (year_max - year_min) + year_min  # de-normalize year
    df_synthetic = pd.DataFrame(synthetic.numpy(), columns=[condition_col] + feature_cols)
    df_synthetic.to_csv(os.path.join(OUTPUT_DATA_PATH, "cvae_synthetic_data.csv"), index=False)

# -------------------------
# Save Model
# -------------------------
torch.save(model.state_dict(), os.path.join(OUTPUT_MODEL_PATH, "cvae_model.pth"))
print("\n✅ CVAE training complete. Future synthetic data and model saved.")
