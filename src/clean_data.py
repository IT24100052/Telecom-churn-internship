import pandas as pd

df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = df["TotalCharges"].str.strip().replace("", "0")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

print("TotalCharges dtype:", df["TotalCharges"].dtype)
print("Blank values remaining:", (df["TotalCharges"].isna()).sum())
print("Number of rows:", len(df))

df.to_csv("data/processed/telco_churn_clean.csv", index=False)

print("Cleaned dataset saved successfully.")