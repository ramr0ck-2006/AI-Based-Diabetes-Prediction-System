from pathlib import Path
import pandas as pd
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
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "diabetes_binary_health.csv"
MODEL_PATH = PROJECT_ROOT / "models"
RESULTS_PATH = PROJECT_ROOT / "results"

RESULTS_PATH.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    # Same preprocessing used during training
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_test, y_test


# ============================================================
# EVALUATE MODELS
# ============================================================

def evaluate_models(X_test, y_test):

    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "Gradient Boosting": "gradient_boosting.pkl",
    }

    results = []

    plt.figure(figsize=(9, 7))

    for model_name, filename in model_files.items():

        model = joblib.load(MODEL_PATH / filename)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.50).astype(int)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )
        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )
        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )
        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        })

        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        cm = confusion_matrix(y_test, predictions)

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Non-Diabetic", "Diabetic"]
        )

        display.plot()
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()

        safe_name = model_name.lower().replace(" ", "_")

        plt.savefig(
            RESULTS_PATH / f"confusion_matrix_{safe_name}.png",
            dpi=150
        )

        plt.close()

        # ----------------------------------------------------
        # ROC Curve
        # ----------------------------------------------------

        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities
        )

        plt.figure(figsize=(8, 6))

        plt.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC = {roc_auc:.3f})"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            RESULTS_PATH / f"roc_curve_{safe_name}.png",
            dpi=150
        )

        plt.close()

        # ----------------------------------------------------
        # Precision-Recall Curve
        # ----------------------------------------------------

        precision_values, recall_values, thresholds = (
            precision_recall_curve(
                y_test,
                probabilities
            )
        )

        plt.figure(figsize=(8, 6))

        plt.plot(
            recall_values,
            precision_values
        )

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(
            f"Precision-Recall Curve - {model_name}"
        )

        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            RESULTS_PATH /
            f"precision_recall_{safe_name}.png",
            dpi=150
        )

        plt.close()

    # --------------------------------------------------------
    # Save evaluation results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_PATH / "detailed_model_evaluation.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("DETAILED MODEL EVALUATION")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def threshold_analysis(X_test, y_test):

    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "Gradient Boosting": "gradient_boosting.pkl",
    }

    threshold_results = []

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]

    for model_name, filename in model_files.items():

        model = joblib.load(MODEL_PATH / filename)

        probabilities = model.predict_proba(X_test)[:, 1]

        for threshold in thresholds:

            predictions = (
                probabilities >= threshold
            ).astype(int)

            precision = precision_score(
                y_test,
                predictions,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                zero_division=0
            )

            threshold_results.append({
                "Model": model_name,
                "Threshold": threshold,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1
            })

    threshold_df = pd.DataFrame(threshold_results)

    threshold_df.to_csv(
        RESULTS_PATH / "threshold_analysis.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    for model_name in model_files:

        model_data = threshold_df[
            threshold_df["Model"] == model_name
        ]

        best_row = model_data.loc[
            model_data["F1-Score"].idxmax()
        ]

        print(f"\n{model_name}")
        print(
            f"Best F1 threshold : "
            f"{best_row['Threshold']:.2f}"
        )
        print(
            f"Precision         : "
            f"{best_row['Precision']:.4f}"
        )
        print(
            f"Recall            : "
            f"{best_row['Recall']:.4f}"
        )
        print(
            f"F1-Score          : "
            f"{best_row['F1-Score']:.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI-BASED DIABETES PREDICTION SYSTEM")
    print("DETAILED MODEL EVALUATION")
    print("=" * 70)

    X_test, y_test = load_data()

    print(f"\nEvaluation records: {len(X_test):,}")

    evaluate_models(X_test, y_test)

    threshold_analysis(X_test, y_test)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nGenerated files are available in:")
    print(RESULTS_PATH)


if __name__ == "__main__":
    main()