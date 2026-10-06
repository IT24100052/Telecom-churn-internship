import pandas as pd

df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("=== DATASET SHAPE ===")
print(df.shape)

print("\n=== COLUMN NAMES ===")
print(df.columns)

print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== TOTALCHARGES SAMPLE VALUES ===")
print(df["TotalCharges"].head(10))

print("\n=== TOTALCHARGES BLANK VALUES ===")
print((df["TotalCharges"].str.strip() == "").sum())

print("\n=== DUPLICATE ROWS ===")
print(df.duplicated().sum())

print("\n=== MISSING VALUES ===")
print(df.isna().sum())


print("\n=== ROWS WITH BLANK TOTALCHARGES ===")
blank_total = df[df["TotalCharges"].str.strip() == ""]

print(
    blank_total[
        [
            "customerID",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Contract",
            "Churn",
        ]
    ]
)

print("\n=== TENURE FOR BLANK TOTALCHARGES ===")
print(blank_total["tenure"].value_counts(dropna=False))