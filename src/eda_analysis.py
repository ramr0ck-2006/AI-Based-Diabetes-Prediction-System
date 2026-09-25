from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "diabetes_binary_health.csv"
RESULTS_PATH = PROJECT_ROOT / "results"

RESULTS_PATH.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Target distribution
    target_counts = df["Diabetes_binary"].value_counts().sort_index()

    print("\nTarget Distribution:")
    print(target_counts)

    # Save target distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Diabetes_binary")
    plt.title("Diabetes Class Distribution")
    plt.xlabel("Diabetes (0 = No, 1 = Yes)")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "target_distribution.png")
    plt.close()

    # Correlation matrix
    correlation = df.corr(numeric_only=True)

    plt.figure(figsize=(14, 11))
    sns.heatmap(correlation, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "correlation_heatmap.png")
    plt.close()

    # Diabetes rate by BMI
    bmi_diabetes = df.groupby("BMI")["Diabetes_binary"].mean()

    plt.figure(figsize=(10, 5))
    bmi_diabetes.plot()
    plt.title("Diabetes Rate by BMI")
    plt.xlabel("BMI")
    plt.ylabel("Diabetes Rate")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "diabetes_by_bmi.png")
    plt.close()

    # Diabetes rate by Age category
    age_diabetes = df.groupby("Age")["Diabetes_binary"].mean()

    plt.figure(figsize=(10, 5))
    age_diabetes.plot(marker="o")
    plt.title("Diabetes Rate by Age Category")
    plt.xlabel("Age Category")
    plt.ylabel("Diabetes Rate")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "diabetes_by_age.png")
    plt.close()

    print("\nEDA completed successfully.")
    print(f"Charts saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()