import os
import pandas as pd

input_dir = "../data/processed"
output_path = "../data/processed/merged_data.csv"

merged_df = pd.DataFrame()

for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        try:
            path = os.path.join(input_dir, file)
            df = pd.read_csv(path)
            print(f"📄 Processing {file}")

            # Check and fix missing 'Indicator Name'
            if 'Indicator Name' not in df.columns:
                print(f"⚠️  'Indicator Name' missing in {file}. Inserting placeholder.")
                df["Indicator Name"] = os.path.splitext(file)[0]  # use filename as indicator

            # Define id_vars (non-year columns)
            id_vars = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
            id_vars_present = [col for col in id_vars if col in df.columns]
            value_vars = [col for col in df.columns if col not in id_vars_present]

            df_melted = df.melt(id_vars=id_vars_present, value_vars=value_vars,
                                var_name='Year', value_name='Value')
            df_melted["Source File"] = file  # optional: track origin
            merged_df = pd.concat([merged_df, df_melted], ignore_index=True)

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

# Save merged file
merged_df.to_csv(output_path, index=False)
print(f"\n✅ Merged data saved to {output_path}")
