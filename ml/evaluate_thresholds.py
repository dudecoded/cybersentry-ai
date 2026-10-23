from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "random_forest.joblib"
TEST_PATH = ROOT_DIR / "dataset" / "UNSW_NB15_testing-set.csv"

THRESHOLDS = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

df = pd.read_csv(TEST_PATH)

X_test = df.drop(columns=["id", "attack_cat", "label"])
y_test = df["label"].astype(int)

model = joblib.load(MODEL_PATH)

print("Generating probabilities...")
probabilities = model.predict_proba(X_test)[:, 1]

results = []

for threshold in THRESHOLDS:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, predictions),
        "attack_precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "attack_recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "attack_f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "false_positive_rate": fp / (fp + tn),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "roc_auc": roc_auc_score(y_test, probabilities),
    })

results_df = pd.DataFrame(results)

output_path = ROOT_DIR / "reports" / "threshold_comparison.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_path, index=False)

print("\nTHRESHOLD COMPARISON")
print(
    results_df[
        [
            "threshold",
            "accuracy",
            "attack_precision",
            "attack_recall",
            "attack_f1",
            "false_positive_rate",
            "false_positives",
            "false_negatives",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

print(f"\nResults saved to: {output_path}")