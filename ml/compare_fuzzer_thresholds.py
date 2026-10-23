from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT_DIR / "dataset" / "UNSW_NB15_testing-set.csv"
MODEL_PATH = ROOT_DIR / "models" / "random_forest.joblib"

BASELINE = 0.50
THRESHOLDS = [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.10]
BATCH_SIZE = 2000


def main():
    print("Loading test dataset...")
    df = pd.read_csv(TEST_PATH)

    feature_columns = [
        col for col in df.columns
        if col not in {"id", "attack_cat", "label"}
    ]

    X = df[feature_columns]
    y = df["label"].to_numpy()
    categories = df["attack_cat"].to_numpy()

    print("Loading production model...")
    model = joblib.load(MODEL_PATH)

    probabilities = np.empty(len(df), dtype=np.float32)

    print("Calculating probabilities in batches...")
    for start in range(0, len(df), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(df))
        probabilities[start:end] = model.predict_proba(
            X.iloc[start:end]
        )[:, 1]
        print(f"Processed {end:,}/{len(df):,}")

    baseline_predictions = probabilities >= BASELINE

    print("\n=== Fuzzer Detection vs False Alarms ===")
    print(
        "Threshold | Fuzzers recovered | Additional normal false alarms "
        "| Total normal false alarms"
    )

    normal_mask = y == 0
    fuzzer_mask = categories == "Fuzzers"

    baseline_fp = int((normal_mask & baseline_predictions).sum())

    for threshold in THRESHOLDS:
        predictions = probabilities >= threshold

        recovered_fuzzers = int(
            (
                fuzzer_mask
                & (y == 1)
                & ~baseline_predictions
                & predictions
            ).sum()
        )

        additional_fp = int(
            (
                normal_mask
                & ~baseline_predictions
                & predictions
            ).sum()
        )

        total_fp = int((normal_mask & predictions).sum())

        print(
            f"{threshold:>9.2f} | "
            f"{recovered_fuzzers:>17,} | "
            f"{additional_fp:>29,} | "
            f"{total_fp:>27,}"
        )

    print(f"\nBaseline threshold: {BASELINE:.2f}")
    print(f"Existing normal false alarms: {baseline_fp:,}")
    print("No model files were modified.")


if __name__ == "__main__":
    main()