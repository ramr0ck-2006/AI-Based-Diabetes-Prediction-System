from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "diabetes_binary_health.csv"


def main():
    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Remove exact duplicate records
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    print(f"\nOriginal records : {before:,}")
    print(f"After duplicates : {after:,}")
    print(f"Removed          : {before - after:,}")

    # Separate features and target
    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"]

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain/Test Split")
    print("-" * 60)
    print(f"Training samples : {len(X_train):,}")
    print(f"Testing samples  : {len(X_test):,}")

    print("\nTraining target distribution:")
    print(y_train.value_counts(normalize=True).mul(100).round(2))

    print("\nTesting target distribution:")
    print(y_test.value_counts(normalize=True).mul(100).round(2))

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()