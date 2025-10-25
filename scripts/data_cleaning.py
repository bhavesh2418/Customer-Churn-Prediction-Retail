# scripts/data_cleaning.py

import pandas as pd
import os

def clean_data(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path)
    print(f"Initial shape: {df.shape}")

    # 1️⃣ Remove duplicates
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")

    # 2️⃣ Handle missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        df = df.dropna()  # or df.fillna(...) depending on context
    print(f"After handling missing values: {df.shape}")

    # 3️⃣ Fix column names (strip whitespace, lowercase)
    df.columns = df.columns.str.strip()

    # 4️⃣ Ensure consistent data types
    bool_cols = ['Email_Opt_In', 'Target_Churn']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to {output_path}")

# Run script
if __name__ == "__main__":
    input_path = r"F:\DataSetKaggle\online_retail_customer_churn.csv"
    output_path = r"F:\CustomerChurnProject\data\cleaned_data.csv"
    clean_data(input_path, output_path)
