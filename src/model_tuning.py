from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "diabetes_binary_health.csv"
MODEL_PATH = PROJECT_ROOT / "models"
RESULTS_PATH = PROJECT_ROOT / "results"

MODEL_PATH.mkdir(exist_ok=True)
RESULTS_PATH.mkdir(exist_ok=True)


def evaluate_model(name, model, X_test, y_test):

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_test, probabilities
        ),
    }


def main():

    print("=" * 70)
    print("MODEL HYPERPARAMETER TUNING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load and prepare data
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    print("\n" + "=" * 70)
    print("TUNING RANDOM FOREST")
    print("=" * 70)

    rf = RandomForestClassifier(
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    rf_params = {
        "n_estimators": [200, 300, 400],
        "max_depth": [None, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_params,
        n_iter=10,
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )

    rf_search.fit(X_train, y_train)

    best_rf = rf_search.best_estimator_

    print("\nBest Random Forest parameters:")
    print(rf_search.best_params_)

    print(
        f"\nBest CV ROC-AUC: "
        f"{rf_search.best_score_:.4f}"
    )

    rf_result = evaluate_model(
        "Tuned Random Forest",
        best_rf,
        X_test,
        y_test,
    )

    print("\nTest Set Results:")
    for key, value in rf_result.items():
        if key != "Model":
            print(f"{key}: {value:.4f}")

    joblib.dump(
        best_rf,
        MODEL_PATH / "tuned_random_forest.pkl"
    )

    # ========================================================
    # GRADIENT BOOSTING
    # ========================================================

    print("\n" + "=" * 70)
    print("TUNING GRADIENT BOOSTING")
    print("=" * 70)

    gb = GradientBoostingClassifier(
        random_state=42
    )

    gb_params = {
        "n_estimators": [100, 150, 200, 250],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "max_depth": [2, 3, 4],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [0.8, 1.0],
    }

    gb_search = RandomizedSearchCV(
        estimator=gb,
        param_distributions=gb_params,
        n_iter=10,
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )

    gb_search.fit(X_train, y_train)

    best_gb = gb_search.best_estimator_

    print("\nBest Gradient Boosting parameters:")
    print(gb_search.best_params_)

    print(
        f"\nBest CV ROC-AUC: "
        f"{gb_search.best_score_:.4f}"
    )

    gb_result = evaluate_model(
        "Tuned Gradient Boosting",
        best_gb,
        X_test,
        y_test,
    )

    print("\nTest Set Results:")
    for key, value in gb_result.items():
        if key != "Model":
            print(f"{key}: {value:.4f}")

    joblib.dump(
        best_gb,
        MODEL_PATH / "tuned_gradient_boosting.pkl"
    )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    comparison = pd.DataFrame([
        rf_result,
        gb_result,
    ])

    comparison.to_csv(
        RESULTS_PATH / "tuned_model_comparison.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("TUNING COMPLETED")
    print("=" * 70)

    print("\nTuned models saved:")
    print("models/tuned_random_forest.pkl")
    print("models/tuned_gradient_boosting.pkl")

    print("\nResults saved:")
    print("results/tuned_model_comparison.csv")


if __name__ == "__main__":
    main()