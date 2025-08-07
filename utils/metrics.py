# metrics.py

import pandas as pd
import os

def load_data(path="../data/processed/merged_data.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["Value"])
    df["Year"] = df["Year"].astype(int)
    return df

def calculate_country_summary(df):
    summary = df.groupby(["Country Name", "Indicator Name"]).agg(
        average_value=("Value", "mean"),
        min_value=("Value", "min"),
        max_value=("Value", "max"),
        std_deviation=("Value", "std")  # ESG Volatility
    ).reset_index()
    return summary

def calculate_trend_slope(df):
    slopes = []
    for (country, indicator), group in df.groupby(["Country Name", "Indicator Name"]):
        group = group.sort_values("Year")
        if len(group["Year"].unique()) >= 2:
            x = group["Year"]
            y = group["Value"]
            slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean())**2).sum()
            slopes.append({
                "Country Name": country,
                "Indicator Name": indicator,
                "Trend Slope": slope
            })
    return pd.DataFrame(slopes)

def calculate_yoy_change(df):
    df = df.sort_values(["Country Name", "Indicator Name", "Year"])
    df["YoY Change"] = df.groupby(["Country Name", "Indicator Name"])["Value"].diff()
    return df

def calculate_z_score(df):
    df["Z-Score"] = df.groupby(["Indicator Name", "Year"])["Value"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )
    return df

def save_metrics(summary_df, slope_df, yoy_df, zscore_df, out_dir="../results/metrics"):
    os.makedirs(out_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(out_dir, "country_summary.csv"), index=False)
    slope_df.to_csv(os.path.join(out_dir, "trend_slopes.csv"), index=False)
    yoy_df.to_csv(os.path.join(out_dir, "yoy_changes.csv"), index=False)
    zscore_df.to_csv(os.path.join(out_dir, "zscore_normalized.csv"), index=False)
    print(f"✅ All metrics saved to {out_dir}")

def run_all():
    print("📥 Loading data...")
    df = load_data()
    print("📊 Calculating summary statistics & volatility...")
    summary = calculate_country_summary(df)
    print("📈 Calculating trend slopes...")
    slopes = calculate_trend_slope(df)
    print("🔁 Calculating YoY changes...")
    yoy = calculate_yoy_change(df)
    print("⚖️  Calculating Z-score normalization...")
    zscore = calculate_z_score(df)
    print("💾 Saving outputs...")
    save_metrics(summary, slopes, yoy, zscore)
    print("🚀 Metrics computation complete!")

if __name__ == "__main__":
    run_all()
