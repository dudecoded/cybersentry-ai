from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT_DIR / "dataset" / "UNSW_NB15_training-set.csv"


def main():
    print("Loading training dataset...")

    df = pd.read_csv(TRAIN_PATH)

    X = df.drop(columns=["id", "attack_cat", "label"])
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training Random Forest on training split...")

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    print("Calculating validation probabilities...")

    probabilities = model.predict_proba(X_val)[:, 1]

    print(f"Validation ROC-AUC: {roc_auc_score(y_val, probabilities):.4f}")

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    print("\n" + "=" * 85)
    print("RANDOM FOREST - VALIDATION THRESHOLD COMPARISON")
    print("=" * 85)

    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1],
        ).ravel()

        results.append(
            {
                "Threshold": threshold,
                "Precision": precision_score(
                    y_val, predictions, zero_division=0
                ),
                "Recall": recall_score(
                    y_val, predictions, zero_division=0
                ),
                "F1-score": f1_score(
                    y_val, predictions, zero_division=0
                ),
                "False Positives": fp,
                "False Negatives": fn,
                "False Positive Rate": fp / (fp + tn),
            }
        )

    results_df = pd.DataFrame(results).set_index("Threshold")
    print(results_df.round(4).to_string())


if __name__ == "__main__":
    main()