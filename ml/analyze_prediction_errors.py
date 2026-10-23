
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT_DIR = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT_DIR / "dataset" / "UNSW_NB15_testing-set.csv"
MODEL_PATH = ROOT_DIR / "models" / "random_forest.joblib"
REPORT_DIR = ROOT_DIR / "reports"

THRESHOLD = 0.50
BATCH_SIZE = 2000


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading official test dataset...")
    df = pd.read_csv(TEST_PATH)

    feature_columns = [
        col for col in df.columns
        if col not in {"id", "attack_cat", "label"}
    ]

    y_test = df["label"].to_numpy()
    probabilities = np.empty(len(df), dtype=np.float32)

    print("Loading production model...")
    model = joblib.load(MODEL_PATH)

    print(f"Generating predictions in batches of {BATCH_SIZE}...")

    for start in range(0, len(df), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(df))

        batch = df.iloc[start:end][feature_columns]
        probabilities[start:end] = model.predict_proba(batch)[:, 1]

        print(f"Processed {end:,}/{len(df):,} records")

    predictions = (probabilities >= THRESHOLD).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print("\n=== Overall Results ===")
    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"True negatives: {tn}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True positives: {tp}")

    results = df[["id", "attack_cat", "label"]].copy()
    results["attack_probability"] = probabilities
    results["prediction"] = predictions

    results["error_type"] = "Correct"

    results.loc[
        (results["label"] == 0) & (results["prediction"] == 1),
        "error_type",
    ] = "False Positive"

    results.loc[
        (results["label"] == 1) & (results["prediction"] == 0),
        "error_type",
    ] = "False Negative"

    category_summary = (
        results.groupby("attack_cat", dropna=False)
        .agg(
            records=("label", "size"),
            actual_attacks=("label", "sum"),
            predicted_attacks=("prediction", "sum"),
            false_positives=(
                "error_type",
                lambda s: (s == "False Positive").sum(),
            ),
            false_negatives=(
                "error_type",
                lambda s: (s == "False Negative").sum(),
            ),
        )
        .reset_index()
        .sort_values("false_negatives", ascending=False)
    )

    normal_records = (
        category_summary["records"]
        - category_summary["actual_attacks"]
    )

    category_summary["false_positive_rate"] = (
        category_summary["false_positives"]
        / normal_records.replace(0, np.nan)
    ).fillna(0)

    category_summary["attack_miss_rate"] = (
        category_summary["false_negatives"]
        / category_summary["actual_attacks"].replace(0, np.nan)
    ).fillna(0)

    errors = results[results["error_type"] != "Correct"]

    category_path = REPORT_DIR / "error_analysis_by_category.csv"
    errors_path = REPORT_DIR / "misclassified_records.csv"

    category_summary.to_csv(category_path, index=False)
    errors.to_csv(errors_path, index=False)

    print("\n=== Errors by Attack Category ===")
    print(
        category_summary[
            [
                "attack_cat",
                "records",
                "false_positives",
                "false_negatives",
                "false_positive_rate",
                "attack_miss_rate",
            ]
        ].to_string(index=False)
    )

    print("\nSaved reports:")
    print(category_path.relative_to(ROOT_DIR))
    print(errors_path.relative_to(ROOT_DIR))


if __name__ == "__main__":
    main()
