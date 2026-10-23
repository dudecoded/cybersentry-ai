from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT_DIR / "dataset" / "UNSW_NB15_training-set.csv"
TEST_PATH = ROOT_DIR / "dataset" / "UNSW_NB15_testing-set.csv"

CATEGORICAL_FEATURES = ["proto", "service", "state"]
EXCLUDED_COLUMNS = ["id", "attack_cat", "label"]

def main():

 print("Loading datasets...")


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 1. Dataset overview
print("\n" + "=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)
print("Columns match:", list(train_df.columns) == list(test_df.columns))

# 2. Label distribution
print("\nLABEL DISTRIBUTION")

for name, df in [("Training", train_df), ("Testing", test_df)]:
    print("\n" + name)

    counts = df["label"].value_counts().sort_index()

    for label, count in counts.items():
        percentage = count / len(df) * 100
        class_name = "Normal" if label == 0 else "Attack"
        print(f"{class_name} ({label}): {count:,} ({percentage:.2f}%)")

# 3. Missing values
print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

for name, df in [("Training", train_df), ("Testing", test_df)]:
    total = int(df.isnull().sum().sum())
    print(f"{name}: {total} missing values")

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        print(missing.to_string())

# 4. Duplicate rows, excluding ID
print("\n" + "=" * 60)
print("DUPLICATE ANALYSIS")
print("=" * 60)

train_compare = train_df.drop(columns=["id"], errors="ignore")
test_compare = test_df.drop(columns=["id"], errors="ignore")

print("Training duplicate rows:", int(train_compare.duplicated().sum()))
print("Testing duplicate rows:", int(test_compare.duplicated().sum()))

common_columns = [
    column
    for column in train_compare.columns
    if column in test_compare.columns
]

train_unique = train_compare[common_columns].drop_duplicates()

test_matches = test_compare[common_columns].merge(
    train_unique,
    on=common_columns,
    how="inner",
)

print("Test rows matching training rows:", len(test_matches))

# 5. Numerical feature comparison
print("\n" + "=" * 60)
print("NUMERICAL FEATURE DIFFERENCES")
print("=" * 60)

numeric_columns = [
    column
    for column in train_df.select_dtypes(include="number").columns
    if column not in EXCLUDED_COLUMNS
]

train_numeric = train_df[numeric_columns]
test_numeric = test_df[numeric_columns]

train_mean = train_numeric.mean()
test_mean = test_numeric.mean()
train_std = train_numeric.std().replace(0, float("nan"))

numerical_results = pd.DataFrame(
    {
        "Train Mean": train_mean,
        "Test Mean": test_mean,
        "Train Std": train_std,
        "Test Std": test_numeric.std(),
    }
)

numerical_results["Standardized Mean Difference"] = (
    numerical_results["Test Mean"] - numerical_results["Train Mean"]
) / numerical_results["Train Std"]

numerical_results["Absolute Difference"] = (
    numerical_results["Standardized Mean Difference"].abs()
)

numerical_results = numerical_results.sort_values(
    "Absolute Difference",
    ascending=False,
    na_position="last",
)

print(numerical_results.head(15).round(4).to_string())

# 6. Categorical feature comparison
print("\n" + "=" * 60)
print("CATEGORICAL FEATURE DIFFERENCES")
print("=" * 60)

for feature in CATEGORICAL_FEATURES:
    print(f"\nFeature: {feature}")

    train_proportions = train_df[feature].value_counts(
        normalize=True,
        dropna=False,
    )

    test_proportions = test_df[feature].value_counts(
        normalize=True,
        dropna=False,
    )

    comparison = pd.concat(
        [
            train_proportions.rename("Train Proportion"),
            test_proportions.rename("Test Proportion"),
        ],
        axis=1,
    ).fillna(0)

    comparison["Absolute Difference"] = (
        comparison["Train Proportion"] - comparison["Test Proportion"]
    ).abs()

    comparison = comparison.sort_values(
        "Absolute Difference",
        ascending=False,
    )

    print(comparison.head(10).round(4).to_string())

print("\n" + "=" * 60)
print("DATA DIAGNOSTICS COMPLETED")
print("=" * 60)


if __name__ == "__main__":
    main()
