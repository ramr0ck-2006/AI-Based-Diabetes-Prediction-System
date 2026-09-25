import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    auc
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/diabetes_binary_health.csv"

RF_MODEL_PATH = "models/tuned_random_forest.pkl"
GB_MODEL_PATH = "models/tuned_gradient_boosting.pkl"

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

print("=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"\nOriginal dataset shape: {df.shape}")

# Remove exact duplicates
df = df.drop_duplicates().reset_index(drop=True)

print(f"After duplicate removal: {df.shape}")

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples : {len(X_test):,}")


# ============================================================
# LOAD TUNED MODELS
# ============================================================

rf_model = joblib.load(RF_MODEL_PATH)
gb_model = joblib.load(GB_MODEL_PATH)

models = {
    "Tuned Random Forest": rf_model,
    "Tuned Gradient Boosting": gb_model
}


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

thresholds = np.arange(0.20, 0.71, 0.05)

threshold_results = []

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

for model_name, model in models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    best_f1 = -1
    best_threshold = 0.50

    print(f"\n{model_name}")

    for threshold in thresholds:

        predictions = (probabilities >= threshold).astype(int)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        threshold_results.append({
            "Model": model_name,
            "Threshold": round(float(threshold), 2),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Best F1 threshold: {best_threshold:.2f}")
    print(f"Best F1 score     : {best_f1:.4f}")


threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    os.path.join(RESULTS_DIR, "final_threshold_analysis.csv"),
    index=False
)


# ============================================================
# FINAL METRICS AT BEST F1 THRESHOLD
# ============================================================

final_results = []

print("\n" + "=" * 70)
print("FINAL MODEL METRICS")
print("=" * 70)

for model_name, model in models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    model_thresholds = threshold_df[
        threshold_df["Model"] == model_name
    ]

    best_row = model_thresholds.loc[
        model_thresholds["F1"].idxmax()
    ]

    threshold = float(best_row["Threshold"])

    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, probabilities)

    final_results.append({
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    })

    print(f"\n{model_name}")
    print(f"Threshold : {threshold:.2f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")


final_df = pd.DataFrame(final_results)

final_df.to_csv(
    os.path.join(RESULTS_DIR, "final_model_comparison.csv"),
    index=False
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

for model_name, model in models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    model_thresholds = threshold_df[
        threshold_df["Model"] == model_name
    ]

    best_threshold = float(
        model_thresholds.loc[
            model_thresholds["F1"].idxmax(),
            "Threshold"
        ]
    )

    predictions = (probabilities >= best_threshold).astype(int)

    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Diabetic", "Diabetic"]
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    disp.plot(ax=ax, cmap="Blues", values_format="d")

    ax.set_title(
        f"{model_name}\nThreshold = {best_threshold:.2f}"
    )

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "")
        + "_final_confusion_matrix.png"
    )

    plt.savefig(
        os.path.join(RESULTS_DIR, filename),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# ROC CURVE
# ============================================================

plt.figure(figsize=(8, 6))

for model_name, model in models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {roc_auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Final Model ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "final_roc_curve.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

plt.figure(figsize=(8, 6))

for model_name, model in models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities
    )

    pr_auc = auc(recall, precision)

    plt.plot(
        recall,
        precision,
        label=f"{model_name} (AUC = {pr_auc:.3f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Final Model Precision-Recall Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "final_precision_recall_curve.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print("✓ results/final_threshold_analysis.csv")
print("✓ results/final_model_comparison.csv")
print("✓ results/final_roc_curve.png")
print("✓ results/final_precision_recall_curve.png")
print("✓ Final confusion matrix images")
