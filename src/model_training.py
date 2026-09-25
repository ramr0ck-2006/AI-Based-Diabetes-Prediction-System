from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "diabetes_binary_health.csv"
MODEL_PATH = PROJECT_ROOT / "models"
RESULTS_PATH = PROJECT_ROOT / "results"

MODEL_PATH.mkdir(exist_ok=True)
RESULTS_PATH.mkdir(exist_ok=True)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):

    print("\n" + "=" * 60)
    print(f"TRAINING: {name}")
    print("=" * 60)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
    }, model


def main():

    print("=" * 60)
    print("AI-BASED DIABETES PREDICTION SYSTEM")
    print("MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    # Remove exact duplicate records
    df = df.drop_duplicates().reset_index(drop=True)

    # Separate features and target
    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"]

    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining records: {len(X_train):,}")
    print(f"Testing records : {len(X_test):,}")

    # --------------------------------------------------------
    # Model 1: Logistic Regression
    # --------------------------------------------------------

    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    # --------------------------------------------------------
    # Model 2: Random Forest
    # --------------------------------------------------------

    random_forest_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    # --------------------------------------------------------
    # Model 3: Gradient Boosting
    # --------------------------------------------------------

    gradient_boosting_model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    # --------------------------------------------------------
    # Train and evaluate
    # --------------------------------------------------------

    results = []

    result, logistic_model = evaluate_model(
        "Logistic Regression",
        logistic_model,
        X_train,
        X_test,
        y_train,
        y_test,
    )
    results.append(result)

    result, random_forest_model = evaluate_model(
        "Random Forest",
        random_forest_model,
        X_train,
        X_test,
        y_train,
        y_test,
    )
    results.append(result)

    result, gradient_boosting_model = evaluate_model(
        "Gradient Boosting",
        gradient_boosting_model,
        X_train,
        X_test,
        y_train,
        y_test,
    )
    results.append(result)

    # --------------------------------------------------------
    # Results table
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="ROC-AUC",
        ascending=False
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(results_df.to_string(index=False))

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df.to_csv(
        RESULTS_PATH / "model_comparison.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save models
    # --------------------------------------------------------

    joblib.dump(
        logistic_model,
        MODEL_PATH / "logistic_regression.pkl"
    )

    joblib.dump(
        random_forest_model,
        MODEL_PATH / "random_forest.pkl"
    )

    joblib.dump(
        gradient_boosting_model,
        MODEL_PATH / "gradient_boosting.pkl"
    )

    print("\nModels saved successfully.")
    print(f"Models folder : {MODEL_PATH}")
    print(f"Results file  : {RESULTS_PATH / 'model_comparison.csv'}")

    print("\nModel training completed successfully.")


if __name__ == "__main__":
    main()