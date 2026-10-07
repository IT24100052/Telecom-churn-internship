import pandas as pd
import sqlite3

df = pd.read_csv("data/processed/telco_churn_clean.csv")

connection = sqlite3.connect("data/processed/telecom_churn.db")

df.to_sql(
    "customers",
    connection,
    if_exists="replace",
    index=False
)

connection.close()

print("Database created successfully.")
print(f"Rows inserted: {len(df)}")