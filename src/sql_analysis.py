import sqlite3
import pandas as pd

connection = sqlite3.connect("data/processed/telecom_churn.db")

query = """
SELECT
    Contract,
    COUNT(*) AS customer_count,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS churn_rate
FROM customers
GROUP BY Contract
ORDER BY churn_rate DESC;
"""

result = pd.read_sql_query(query, connection)

print("=== CHURN BY CONTRACT ===")
print(result.to_string(index=False))

query2 = """
SELECT
    customerID,
    tenure,
    MonthlyCharges,
    TotalCharges,
    Contract,
    Churn
FROM customers
WHERE MonthlyCharges > 80
  AND Contract = 'Month-to-month'
  AND Churn = 'Yes'
ORDER BY MonthlyCharges DESC
LIMIT 10;
"""

result2 = pd.read_sql_query(query2, connection)

print("\n=== HIGH-VALUE CHURNED MONTH-TO-MONTH CUSTOMERS ===")
print(result2.to_string(index=False))

query3 = """
SELECT
    PaymentMethod,
    COUNT(*) AS customer_count,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS churn_rate
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate DESC;
"""

result3 = pd.read_sql_query(query3, connection)

print("\n=== CHURN BY PAYMENT METHOD ===")
print(result3.to_string(index=False))

query4 = """
SELECT
    CASE
        WHEN tenure <= 12 THEN '0-12 months'
        WHEN tenure > 48 THEN '49+ months'
        ELSE 'Other'
    END AS tenure_group,
    COUNT(*) AS customer_count,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS churn_rate
FROM customers
GROUP BY tenure_group
ORDER BY churn_rate DESC;
"""

result4 = pd.read_sql_query(query4, connection)

print("\n=== CHURN BY TENURE GROUP ===")
print(result4.to_string(index=False))

connection.close()