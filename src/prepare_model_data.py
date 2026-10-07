import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv("data/processed/telco_churn_clean.csv")


# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

X = df.drop(columns=["customerID", "Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# --------------------------------------------------
# 3. Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("=== DATA SPLIT ===")
print("Full dataset:", X.shape)
print("Training set:", X_train.shape)
print("Test set:", X_test.shape)


print("\n=== TARGET DISTRIBUTION ===")
print("Full churn rate:", round(y.mean() * 100, 2), "%")
print("Train churn rate:", round(y_train.mean() * 100, 2), "%")
print("Test churn rate:", round(y_test.mean() * 100, 2), "%")


# --------------------------------------------------
# 4. Identify numeric and categorical features
# --------------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()


print("\n=== NUMERIC FEATURES ===")
print(numeric_features)

print("\n=== CATEGORICAL FEATURES ===")
print(categorical_features)

print("\nNumeric count:", len(numeric_features))
print("Categorical count:", len(categorical_features))


# --------------------------------------------------
# 5. Build preprocessing
# --------------------------------------------------

numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# --------------------------------------------------
# 6. Inspect processed training data
# --------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)

print("\n=== PREPROCESSED TRAINING DATA ===")
print("Original shape:", X_train.shape)
print("Processed shape:", X_train_processed.shape)


# --------------------------------------------------
# 7. Build Logistic Regression pipeline
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)
tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            random_state=42,
            max_depth=5
        ))
    ]
)

tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)
tree_prob = tree_model.predict_proba(X_test)[:, 1]

print("\n=== DECISION TREE ===")
print("Accuracy:", round(accuracy_score(y_test, tree_pred), 4))
print("Precision:", round(precision_score(y_test, tree_pred), 4))
print("Recall:", round(recall_score(y_test, tree_pred), 4))
print("F1-score:", round(f1_score(y_test, tree_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, tree_prob), 4))

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            max_depth=8,
            class_weight="balanced"
        ))
    ]
)

random_forest_model.fit(X_train, y_train)

rf_pred = random_forest_model.predict(X_test)
rf_prob = random_forest_model.predict_proba(X_test)[:, 1]

print("\n=== RANDOM FOREST ===")
print("Accuracy:", round(accuracy_score(y_test, rf_pred), 4))
print("Precision:", round(precision_score(y_test, rf_pred), 4))
print("Recall:", round(recall_score(y_test, rf_pred), 4))
print("F1-score:", round(f1_score(y_test, rf_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, rf_prob), 4))

# --------------------------------------------------
# 8. Train model
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# 9. Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 10. Evaluate baseline model
# --------------------------------------------------

print("\n=== LOGISTIC REGRESSION BASELINE ===")

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1-score:", round(f1_score(y_test, y_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))


# --------------------------------------------------
# 11. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\n=== CONFUSION MATRIX ===")
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

print("\n=== THRESHOLD COMPARISON ===")

thresholds = [0.30, 0.40, 0.50, 0.60]

for threshold in thresholds:
    threshold_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_test, threshold_pred)
    recall = recall_score(y_test, threshold_pred)
    f1 = f1_score(y_test, threshold_pred)

    print(
        f"Threshold {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )

    print("\n=== CONFUSION MATRIX AT THRESHOLD 0.30 ===")

selected_threshold = 0.30

y_pred_030 = (y_prob >= selected_threshold).astype(int)

cm_030 = confusion_matrix(y_test, y_pred_030)

tn_030, fp_030, fn_030, tp_030 = cm_030.ravel()

print("True Negatives:", tn_030)
print("False Positives:", fp_030)
print("False Negatives:", fn_030)
print("True Positives:", tp_030)

rf_cm = confusion_matrix(y_test, rf_pred)

rf_tn, rf_fp, rf_fn, rf_tp = rf_cm.ravel()

print("\n=== RANDOM FOREST CONFUSION MATRIX ===")
print("True Negatives:", rf_tn)
print("False Positives:", rf_fp)
print("False Negatives:", rf_fn)
print("True Positives:", rf_tp)