import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/diabetes_binary_health.csv"
MODEL_PATH = "models/tuned_gradient_boosting.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MODEL PROBABILITY SANITY CHECK")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"\nOriginal dataset: {df.shape}")

# Same preprocessing used during training/evaluation
df = df.drop_duplicates().reset_index(drop=True)

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Test samples: {len(X_test):,}")


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# GET PROBABILITIES
# ============================================================

probabilities = model.predict_proba(X_test)[:, 1]


# ============================================================
# BASIC STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("PROBABILITY DISTRIBUTION")
print("=" * 70)

print(f"\nMinimum probability : {probabilities.min() * 100:.2f}%")
print(f"Maximum probability : {probabilities.max() * 100:.2f}%")
print(f"Mean probability    : {probabilities.mean() * 100:.2f}%")
print(f"Median probability  : {np.median(probabilities) * 100:.2f}%")


# ============================================================
# PERCENTILES
# ============================================================

print("\n" + "=" * 70)
print("PROBABILITY PERCENTILES")
print("=" * 70)

for percentile in [50, 75, 90, 95, 99, 99.5, 99.9]:

    value = np.percentile(
        probabilities,
        percentile
    )

    print(
        f"{percentile:5.1f}th percentile : "
        f"{value * 100:.2f}%"
    )


# ============================================================
# HIGH-PROBABILITY COUNTS
# ============================================================

print("\n" + "=" * 70)
print("HIGH-PROBABILITY PREDICTIONS")
print("=" * 70)

for threshold in [0.25, 0.50, 0.60, 0.70, 0.80, 0.90]:

    count = np.sum(
        probabilities >= threshold
    )

    percentage = (
        count / len(probabilities)
    ) * 100

    print(
        f">= {threshold * 100:5.0f}% : "
        f"{count:6,} samples "
        f"({percentage:.3f}%)"
    )


# ============================================================
# TOP 20 HIGHEST PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 HIGHEST MODEL PROBABILITIES")
print("=" * 70)

top_indices = np.argsort(
    probabilities
)[-20:][::-1]

top_data = X_test.iloc[top_indices].copy()

top_data["Actual_Diabetes"] = (
    y_test.iloc[top_indices].values
)

top_data["Predicted_Probability"] = (
    probabilities[top_indices]
)

top_data = top_data[
    [
        "Predicted_Probability",
        "Actual_Diabetes"
    ]
]

top_data["Predicted_Probability"] = (
    top_data["Predicted_Probability"] * 100
)

top_data["Predicted_Probability"] = (
    top_data["Predicted_Probability"].round(2)
)

print(
    top_data.to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (
    "results/probability_sanity_check.csv"
)

full_results = X_test.copy()

full_results["Actual_Diabetes"] = y_test.values

full_results["Predicted_Probability"] = (
    probabilities
)

full_results.to_csv(
    output_path,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("SANITY CHECK COMPLETE")
print("=" * 70)

print(
    f"\nSaved detailed results to:"
    f"\n{output_path}"
)