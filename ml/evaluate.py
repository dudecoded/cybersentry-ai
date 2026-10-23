from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix,
roc_auc_score,
classification_report,
)

# Project paths

ROOT_DIR = Path(__file__).resolve().parent.parent

TEST_PATH = ROOT_DIR / "dataset" / "UNSW_NB15_testing-set.csv"
MODEL_PATH = ROOT_DIR / "models" / "logistic_regression.joblib"

def main():
# 1. Load test dataset
 print("Loading test dataset...")


test_df = pd.read_csv(TEST_PATH)

X_test = test_df.drop(
    columns=["id", "attack_cat", "label"]
)

y_test = test_df["label"]

print("Testing shape:", X_test.shape)

# 2. Load trained model
print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)

# 3. Predict attack probabilities
print("\nCalculating attack probabilities...")

probabilities = model.predict_proba(X_test)[:, 1]

print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

# 4. Compare thresholds
thresholds = [0.40, 0.50]

print("\n" + "=" * 85)
print("CYBERSENTRY AI - THRESHOLD COMPARISON")
print("=" * 85)

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    false_positive_rate = (
        fp / (fp + tn) if (fp + tn) > 0 else 0
    )

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1-score:            {f1:.4f}")
    print(f"False Positive Rate: {false_positive_rate:.4f}")

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )
    )

    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")

# 5. Detailed classification report at threshold 0.50
predictions = (probabilities >= 0.50).astype(int)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT - THRESHOLD 0.50")
print("=" * 60)

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        digits=4,
        zero_division=0,
    )
)


if __name__ == "__main__":
    main()