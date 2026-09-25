from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "diabetes_binary_health.csv"
MODEL_PATH = PROJECT_ROOT / "models"
RESULTS_PATH = PROJECT_ROOT / "results"

RESULTS_PATH.mkdir(exist_ok=True)


def main():

    print("=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=["Diabetes_binary"])

    feature_names = X.columns

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    rf = joblib.load(
        MODEL_PATH / "random_forest.pkl"
    )

    rf_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    print("\nRandom Forest Feature Importance")
    print("-" * 70)
    print(rf_importance.to_string(index=False))

    rf_importance.to_csv(
        RESULTS_PATH / "random_forest_feature_importance.csv",
        index=False
    )

    # Plot
    plt.figure(figsize=(10, 8))

    plt.barh(
        rf_importance["Feature"].head(15)[::-1],
        rf_importance["Importance"].head(15)[::-1]
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Top 15 Feature Importances - Random Forest")
    plt.tight_layout()

    plt.savefig(
        RESULTS_PATH / "random_forest_feature_importance.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Gradient Boosting
    # --------------------------------------------------------

    gb = joblib.load(
        MODEL_PATH / "gradient_boosting.pkl"
    )

    gb_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": gb.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    print("\nGradient Boosting Feature Importance")
    print("-" * 70)
    print(gb_importance.to_string(index=False))

    gb_importance.to_csv(
        RESULTS_PATH / "gradient_boosting_feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        gb_importance["Feature"].head(15)[::-1],
        gb_importance["Importance"].head(15)[::-1]
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Top 15 Feature Importances - Gradient Boosting")
    plt.tight_layout()

    plt.savefig(
        RESULTS_PATH / "gradient_boosting_feature_importance.png",
        dpi=150
    )

    plt.close()

    print("\n" + "=" * 70)
    print("FEATURE ANALYSIS COMPLETED")
    print("=" * 70)

    print(f"\nResults saved to:")
    print(RESULTS_PATH)


if __name__ == "__main__":
    main()