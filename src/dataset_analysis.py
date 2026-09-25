from pathlib import Path
import pandas as pd


# ============================================================
# AI-BASED DIABETES PREDICTION SYSTEM
# Dataset Analysis
# ============================================================

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "diabetes_binary_health.csv"


def main():
    print("=" * 70)
    print("AI-BASED DIABETES PREDICTION SYSTEM")
    print("DATASET ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check dataset path
    # --------------------------------------------------------
    if not DATA_PATH.exists():
        print(f"\nERROR: Dataset not found at:\n{DATA_PATH}")
        return

    print(f"\nDataset path:")
    print(DATA_PATH)

    # --------------------------------------------------------
    # 2. Load dataset
    # --------------------------------------------------------
    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")

    # --------------------------------------------------------
    # 3. Basic shape
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. DATASET SHAPE")
    print("=" * 70)

    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    # --------------------------------------------------------
    # 4. Column names
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. COLUMN NAMES")
    print("=" * 70)

    for index, column in enumerate(df.columns, start=1):
        print(f"{index:2}. {column}")

    # --------------------------------------------------------
    # 5. Data types
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. DATA TYPES")
    print("=" * 70)

    print(df.dtypes)

    # --------------------------------------------------------
    # 6. Missing values
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. MISSING VALUES")
    print("=" * 70)

    missing = df.isnull().sum()

    missing_table = pd.DataFrame({
        "Missing Values": missing,
        "Missing Percentage": (missing / len(df) * 100).round(2)
    })

    print(missing_table)

    # --------------------------------------------------------
    # 7. Duplicate records
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. DUPLICATE RECORDS")
    print("=" * 70)

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_count:,}")

    # --------------------------------------------------------
    # 8. Target distribution
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. TARGET DISTRIBUTION")
    print("=" * 70)

    target = "Diabetes_binary"

    if target in df.columns:
        target_counts = df[target].value_counts().sort_index()
        target_percentages = (
            df[target]
            .value_counts(normalize=True)
            .sort_index()
            .mul(100)
            .round(2)
        )

        target_table = pd.DataFrame({
            "Count": target_counts,
            "Percentage": target_percentages
        })

        print(target_table)

    else:
        print(f"WARNING: Target column '{target}' was not found.")

    # --------------------------------------------------------
    # 9. Unique values
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("7. UNIQUE VALUES PER FEATURE")
    print("=" * 70)

    unique_values = df.nunique().sort_values()

    for column, count in unique_values.items():
        print(f"{column:25} : {count}")

    # --------------------------------------------------------
    # 10. Descriptive statistics
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("8. DESCRIPTIVE STATISTICS")
    print("=" * 70)

    print(df.describe().T)

    # --------------------------------------------------------
    # 11. Potential invalid / negative values
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("9. NEGATIVE VALUES")
    print("=" * 70)

    numeric_columns = df.select_dtypes(include="number").columns

    negative_counts = (df[numeric_columns] < 0).sum()

    negative_counts = negative_counts[negative_counts > 0]

    if negative_counts.empty:
        print("No negative values found.")
    else:
        print(negative_counts)

    # --------------------------------------------------------
    # 12. Sample records
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("10. FIRST 5 RECORDS")
    print("=" * 70)

    print(df.head())

    # --------------------------------------------------------
    # 13. Final summary
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"Rows       : {len(df):,}")
    print(f"Features   : {len(df.columns) - 1}")
    print(f"Target     : {target}")
    print(f"Duplicates : {duplicate_count:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()