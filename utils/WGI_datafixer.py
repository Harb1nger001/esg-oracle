import pandas as pd
import os

# Input and output paths
input_path = r"D:\Quantum ESG forcasting\data\raw\wgidataset.xlsx"
output_path = r"D:\Quantum ESG forcasting\data\processed\wgi_pivoted.csv"

# Load the raw WGI data
df = pd.read_excel(input_path)

# Drop rows where estimate or other essential fields are missing
df = df.dropna(subset=['countryname', 'code', 'indicator', 'year', 'estimate'])

# Replace '..' with NaN and ensure numeric estimate
df['estimate'] = pd.to_numeric(df['estimate'], errors='coerce')

# Now pivot the table
pivoted = df.pivot_table(
    index=['countryname', 'code', 'indicator'],
    columns='year',
    values='estimate',
    aggfunc='mean'  # If there's duplicate year data, just average it
).reset_index()

# Rename and rearrange columns
pivoted.rename(columns={
    'countryname': 'Country Name',
    'code': 'Country Code',
    'indicator': 'Indicator Name'
}, inplace=True)

# Add dummy 'Indicator Code' column (can be updated later)
pivoted.insert(3, 'Indicator Code', 'WGI_DUMMY')

# Order columns: metadata first, then years
year_cols = sorted([col for col in pivoted.columns if isinstance(col, int)])
final_cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'] + year_cols
pivoted = pivoted[final_cols]

# Save to processed folder
os.makedirs(os.path.dirname(output_path), exist_ok=True)
pivoted.to_csv(output_path, index=False)

print("✅ Fixed WGI data saved to:", output_path)
