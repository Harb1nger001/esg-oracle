# data_utils.py

import os
import pandas as pd
from pathlib import Path

def clean_and_save_csv(file_paths, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for path in file_paths:
        df = pd.read_csv(path)

        # Drop irrelevant columns if they exist
        df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True)

        # Replace '..' with NaN
        df.replace('..', pd.NA, inplace=True)

        # Convert all year columns to numeric
        year_cols = [col for col in df.columns if col.isdigit()]
        df[year_cols] = df[year_cols].apply(pd.to_numeric, errors='coerce')

        # Drop rows where all year data is missing
        df.dropna(subset=year_cols, how='all', inplace=True)

        # Forward fill missing values across columns (per row)
        df[year_cols] = df[year_cols].apply(lambda row: row.ffill(axis=0), axis=1)

        # Save processed version
        file_name = Path(path).stem + '_cleaned.csv'
        save_path = save_dir / file_name
        df.to_csv(save_path, index=False)
        print(f"[✓] Saved cleaned data to: {save_path}")

if __name__ == "__main__":
    raw_paths = [
        r"D:\Quantum ESG forcasting\data\raw\API_SL.EMP.TOTL.SP.ZS_DS2_en_csv_v2_22253.csv",
        r"D:\Quantum ESG forcasting\data\raw\API_AG.LND.FRST.ZS_DS2_en_csv_v2_38121.csv",
        r"D:\Quantum ESG forcasting\data\raw\API_EG.FEC.RNEW.ZS_DS2_en_csv_v2_23178.csv",
        r"D:\Quantum ESG forcasting\data\raw\API_EN.ATM.PM25.MC.M3_DS2_en_csv_v2_30486.csv",
        r"D:\Quantum ESG forcasting\data\raw\API_EN.GHG.CO2.RT.GDP.PP.KD_DS2_en_csv_v2_37939.csv",
        r"D:\Quantum ESG forcasting\data\raw\API_EN.GHG.CO2.ZG.AR5_DS2_en_csv_v2_22149.csv",
        r"D:\Quantum ESG forcasting\data\raw\API_PV.EST_DS2_en_csv_v2_31014.csv",
        r"D:\Quantum ESG forcasting\data\raw\API_SE.ADT.1524.LT.ZS_DS2_en_csv_v2_22827.csv",
        r"D:\Quantum ESG forcasting\data\raw\API_SH.XPD.CHEX.PC.CD_DS2_en_csv_v2_22334.csv",
        r"D:\Quantum ESG forcasting\data\processed\wgi_pivoted.csv"
    ]
    
    clean_and_save_csv(raw_paths, r"D:\Quantum ESG forcasting\data\processed")
