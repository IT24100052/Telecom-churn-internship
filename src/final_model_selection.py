import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/telco_churn_clean.csv"
)

X = df.drop(
    columns=["customerID", "Churn"]
)

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# --------------------------------------------------
# 2. Create TRAIN / VALIDATION / FINAL TEST
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)


print("=== FINAL DATA SPLIT ===")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Final test:", X_test.shape)

print("\nChurn rates:")
print(
    "Train:",
    round(y_train.mean() * 100, 2),
    "%"
)
print(
    "Validation:",
    round(y_val.mean() * 100, 2),
    "%"
)
print(
    "Final test:",
    round(y_test.mean() * 100, 2),
    "%"
)


# --------------------------------------------------
# 3. Feature types
# --------------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()


# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features,
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features,
        ),
    ]
)


# --------------------------------------------------
# 5. Logistic Regression
# --------------------------------------------------

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            ),
        ),
    ]
)

logistic_model.fit(
    X_train,
    y_train
)

logistic_prob = logistic_model.predict_proba(
    X_val
)[:, 1]


# --------------------------------------------------
# 6. Logistic Regression threshold comparison
# --------------------------------------------------

print(
    "\n=== LOGISTIC REGRESSION VALIDATION ==="
)

for threshold in [0.30, 0.40, 0.50]:

    pred = (
        logistic_prob >= threshold
    ).astype(int)

    print(
        f"\nThreshold {threshold:.2f}"
    )

    print(
        "Precision:",
        round(
            precision_score(y_val, pred),
            4
        )
    )

    print(
        "Recall:",
        round(
            recall_score(y_val, pred),
            4
        )
    )

    print(
        "F1:",
        round(
            f1_score(y_val, pred),
            4
        )
    )


print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_val,
            logistic_prob
        ),
        4
    )
)


# --------------------------------------------------
# 7. Random Forest
# --------------------------------------------------

rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)


rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(X_val)

rf_prob = rf_model.predict_proba(
    X_val
)[:, 1]


print(
    "\n=== RANDOM FOREST VALIDATION ==="
)

print(
    "Accuracy:",
    round(
        accuracy_score(
            y_val,
            rf_pred
        ),
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(
            y_val,
            rf_pred
        ),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(
            y_val,
            rf_pred
        ),
        4
    )
)

print(
    "F1:",
    round(
        f1_score(
            y_val,
            rf_pred
        ),
        4
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_val,
            rf_prob
        ),
        4
    )
)
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# 8. FINAL TEST EVALUATION
# --------------------------------------------------

final_pred = rf_model.predict(X_test)
final_prob = rf_model.predict_proba(X_test)[:, 1]


print("\n=== FINAL RANDOM FOREST TEST RESULTS ===")

print(
    "Accuracy:",
    round(
        accuracy_score(y_test, final_pred),
        4
    )
)

print(
    "Precision:",
    round(
        precision_score(y_test, final_pred),
        4
    )
)

print(
    "Recall:",
    round(
        recall_score(y_test, final_pred),
        4
    )
)

print(
    "F1:",
    round(
        f1_score(y_test, final_pred),
        4
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(y_test, final_prob),
        4
    )
)


final_cm = confusion_matrix(
    y_test,
    final_pred
)

tn, fp, fn, tp = final_cm.ravel()


print("\n=== FINAL TEST CONFUSION MATRIX ===")
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

# --------------------------------------------------
# 9. Save final model
# --------------------------------------------------

model_path = "models/telecom_churn_model.joblib"

joblib.dump(
    rf_model,
    model_path
)

print("\n=== MODEL SAVED ===")
print("Path:", model_path)