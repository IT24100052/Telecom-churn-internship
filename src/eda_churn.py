import pandas as pd

df = pd.read_csv("data/processed/telco_churn_clean.csv")

print("=== CHURN COUNTS ===")
print(df["Churn"].value_counts())

print("\n=== CHURN PERCENTAGE DISTRIBUTION ===")
print(df["Churn"].value_counts(normalize=True) * 100)

churn_rate = (df["Churn"] == "Yes").mean() * 100

print("\n=== OVERALL CHURN RATE ===")
print(f"{churn_rate:.2f}%")

print("\n=== TOTAL CUSTOMERS ===")
print(len(df))

df["ChurnFlag"] = df["Churn"].map({"No": 0, "Yes": 1})

contract_summary = df.groupby("Contract").agg(
    Customers=("customerID", "count"),
    Churned=("ChurnFlag", "sum"),
    Churn_Rate=("ChurnFlag", "mean")
)

contract_summary["Churn_Rate"] = contract_summary["Churn_Rate"] * 100

print("\n=== CHURN BY CONTRACT TYPE ===")
print(contract_summary)

print("\n=== CHURN BY TENURE GROUP ===")

bins = [0, 12, 24, 48, 72]
labels = ["0-12 months", "13-24 months", "25-48 months", "49-72 months"]

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

tenure_summary = df.groupby("TenureGroup", observed=False).agg(
    Customers=("customerID", "count"),
    Churned=("ChurnFlag", "sum"),
    Churn_Rate=("ChurnFlag", "mean")
)

tenure_summary["Churn_Rate"] = tenure_summary["Churn_Rate"] * 100

print(tenure_summary)

print("\n=== MONTHLY CHARGES BY CHURN ===")

monthly_summary = df.groupby("Churn").agg(
    Customers=("customerID", "count"),
    Avg_Monthly_Charges=("MonthlyCharges", "mean"),
    Median_Monthly_Charges=("MonthlyCharges", "median")
)

print(monthly_summary.to_string())