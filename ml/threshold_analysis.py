from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT_DIR / "dataset" / "UNSW_NB15_training-set.csv"


def main():
    # 1. Load the training dataset
    print("Loading training dataset...")

    df = pd.read_csv(TRAIN_PATH)

    X = df.drop(columns=["id", "attack_cat", "label"])
    y = df["label"]

    # 2. Create a validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))

    # 3. Build a fresh pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # 4. Train on the training portion only
    print("\nTraining model for threshold analysis...")

    model.fit(X_train, y_train)

    # 5. Get attack probabilities for validation data
    print("Calculating validation probabilities...")

    probabilities = model.predict_proba(X_val)[:, 1]

    print(f"Validation ROC-AUC: {roc_auc_score(y_val, probabilities):.4f}")

    # 6. Compare classification thresholds
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1],
        ).ravel()

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0,
        )

        false_positive_rate = fp / (fp + tn) if (fp + tn) else 0

        results.append(
            {
                "Threshold": threshold,
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1 Score": round(f1, 4),
                "False Positive Rate": round(false_positive_rate, 4),
                "False Positives": fp,
                "False Negatives": fn,
            }
        )

    # 7. Display results
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print("CYBERSENTRY AI - THRESHOLD ANALYSIS")
    print("=" * 90)

    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()