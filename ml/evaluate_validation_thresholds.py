
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT_DIR / "dataset" / "UNSW_NB15_training-set.csv"
MODEL_PATH = ROOT_DIR / "models" / "random_forest.joblib"

THRESHOLDS = [0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80]


def main():
    print("Loading training dataset...")
    df = pd.read_csv(TRAIN_PATH)

    X = df.drop(columns=["id", "attack_cat", "label"])
    y = df["label"].astype(int)

    # Recreate a stratified holdout for threshold selection.
    _, X_val, _, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Validation records: {len(X_val):,}")
    print("Loading existing trained model...")
    model = joblib.load(MODEL_PATH)

    print("Generating validation probabilities...")
    probabilities = model.predict_proba(X_val)[:, 1]

    rows = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ).ravel()

        rows.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_val, predictions),
            "attack_precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "attack_recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "false_positive_rate": fp / (fp + tn),
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "true_negatives": tn,
        })

    results = pd.DataFrame(rows)

    report_dir = ROOT_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    output_path = report_dir / "validation_threshold_comparison.csv"
    results.to_csv(output_path, index=False)

    print("\nVALIDATION THRESHOLD COMPARISON")
    print(results.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()

