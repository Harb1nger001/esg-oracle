import pandas as pd

# File paths
INPUT_FILE = r"D:\Quantum ESG forcasting\results\metrics\zscore_normalized.csv"
OUTPUT_FILE = r"D:\Quantum ESG forcasting\results\esg_scores_grouped.csv"

# Load the data
df = pd.read_csv(INPUT_FILE)

# Pivot to wide format using Z-Score
df_wide = df.pivot_table(
    index=["Country Name", "Year"],
    columns="Indicator Name",
    values="Z-Score"
).reset_index()
df_wide.columns.name = None  # Remove pandas' automatic label

# Define indicators per group
E_INDICATORS = [
    "Forest area (% of land area)",
    "Renewable energy consumption (% of total final energy consumption)",
    "PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)",
    "Carbon intensity of GDP (kg CO2e per 2021 PPP $ of GDP)",
    "Carbon dioxide (CO2) emissions (total) excluding LULUCF (% change from 1990)"
]

S_INDICATORS = [
    "Literacy rate, youth total (% of people ages 15-24)",
    "Current health expenditure per capita (current US$)",
    "Employment to population ratio, 15+, total (%) (modeled ILO estimate)"
]

G_INDICATORS = [
    "Political Stability and Absence of Violence/Terrorism: Estimate",
    "wgi_pivoted_cleaned"
]

# Group average function
def compute_group_score(row, indicators):
    values = [row.get(ind) for ind in indicators if ind in row]
    values = [v for v in values if pd.notnull(v)]
    return sum(values) / len(values) if values else None

# Compute E, S, G scores
df_wide['E_Score'] = df_wide.apply(lambda row: compute_group_score(row, E_INDICATORS), axis=1)
df_wide['S_Score'] = df_wide.apply(lambda row: compute_group_score(row, S_INDICATORS), axis=1)
df_wide['G_Score'] = df_wide.apply(lambda row: compute_group_score(row, G_INDICATORS), axis=1)

# Compute ESG score (mean of E, S, G)
df_wide['ESG_Score'] = df_wide[['E_Score', 'S_Score', 'G_Score']].mean(axis=1)

# Save to CSV
df_wide.to_csv(OUTPUT_FILE, index=False)
print(f"✅ ESG scores saved to: {OUTPUT_FILE}")
