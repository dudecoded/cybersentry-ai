
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT_DIR / "dataset" / "UNSW_NB15_training-set.csv"
REPORT_DIR = ROOT_DIR / "reports"
EVALUATION_MODEL_PATH = ROOT_DIR / "models" / "random_forest_validation.joblib"

THRESHOLDS = [0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80]


def main():
    print("Loading training dataset...")
    df = pd.read_csv(TRAIN_PATH)

    X = df.drop(columns=["id", "attack_cat", "label"])
    y = df["label"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training records:   {len(X_train):,}")
    print(f"Validation records: {len(X_val):,}")

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
               RandomForestClassifier(
    n_estimators=50,
    class_weight="balanced",
    random_state=42,
    n_jobs=1,
    max_samples=0.8,
),
            ),
        ]
    )

    print("\nTraining evaluation model on training split only...")
    model.fit(X_train, y_train)

    # Save under a different filename. Do not overwrite production model.
    EVALUATION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, EVALUATION_MODEL_PATH)
    print(f"Evaluation model saved to: {EVALUATION_MODEL_PATH}")

    print("\nPredicting on unseen validation records...")
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
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        })

    results = pd.DataFrame(rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "clean_validation_thresholds.csv"
    results.to_csv(output_path, index=False)

    print("\nCLEAN VALIDATION RESULTS")
    print(results.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    ))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

