from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "random_forest_validation.joblib"
TEST_PATH = ROOT_DIR / "dataset" / "UNSW_NB15_testing-set.csv"

THRESHOLD = 0.65


def main():
    print("Loading evaluation model and official test dataset...")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(TEST_PATH)

    X_test = df.drop(columns=["id", "attack_cat", "label"])
    y_test = df["label"].astype(int)

    print(f"Test records: {len(X_test):,}")
    print("Generating predictions...")

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print("\nOFFICIAL TEST SET RESULTS")
    print(f"Threshold:            {THRESHOLD:.2f}")
    print(f"Accuracy:             {accuracy_score(y_test, predictions):.4f}")
    print(f"ROC-AUC:              {roc_auc_score(y_test, probabilities):.4f}")
    print(f"Attack precision:     {precision_score(y_test, predictions):.4f}")
    print(f"Attack recall:         {recall_score(y_test, predictions):.4f}")
    print(f"False-positive rate:   {fp / (fp + tn):.4f}")
    print(f"True negatives:        {tn:,}")
    print(f"False positives:       {fp:,}")
    print(f"False negatives:       {fn:,}")
    print(f"True positives:        {tp:,}")


if __name__ == "__main__":
    main()