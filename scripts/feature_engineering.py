# scripts/feature_engineering.py

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def feature_engineering(input_path, output_path):
    # 1️⃣ Load cleaned data
    df = pd.read_csv(input_path)
    print(f"Initial shape: {df.shape}")

    # 2️⃣ Rename target column (if not already)
    if 'Churn' in df.columns:
        df.rename(columns={'Churn': 'Target_Churn'}, inplace=True)

    # 3️⃣ Encode categorical variables
    categorical_cols = ['Gender', 'Promotion_Response']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            print(f"Encoded {col}")

    # 4️⃣ Convert boolean columns to integers
    bool_cols = ['Email_Opt_In', 'Target_Churn']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
            print(f"Converted {col} to int")

    # 5️⃣ Optional: Scale numeric features (can skip for tree-based models)
    # numeric_cols = ['Age', 'Annual_Income', 'Total_Spend', 'Years_as_Customer', 
    #                 'Num_of_Purchases', 'Average_Transaction_Amount', 'Num_of_Returns', 
    #                 'Num_of_Support_Contacts', 'Satisfaction_Score', 'Last_Purchase_Days_Ago']
    # scaler = StandardScaler()
    # df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    # print("Scaled numeric features")

    # 6️⃣ Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Feature-engineered data saved to {output_path}")


# Run script
if __name__ == "__main__":
    input_path = r"F:\CustomerChurnProject\data\cleaned_data.csv"
    output_path = r"F:\CustomerChurnProject\data\cleaned_churn_data.csv"
    feature_engineering(input_path, output_path)

