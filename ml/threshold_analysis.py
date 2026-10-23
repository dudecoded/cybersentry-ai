
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT_DIR / "dataset" / "UNSW_NB15_training-set.csv"
RESULTS_PATH = ROOT_DIR / "ml" / "rf_threshold_results.csv"
THRESHOLD_PATH = ROOT_DIR / "models" / "selected_threshold.txt"


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

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))

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

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    print("Calculating validation probabilities...")
    probabilities = model.predict_proba(X_val)[:, 1]

    print(f"Validation ROC-AUC: {roc_auc_score(y_val, probabilities):.4f}")

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1],
        ).ravel()

        precision = precision_score(y_val, predictions, zero_division=0)
        recall = recall_score(y_val, predictions, zero_division=0)
        f1 = f1_score(y_val, predictions, zero_division=0)
        fpr = fp / (fp + tn) if (fp + tn) else 0

        results.append(
            {
                "Threshold": threshold,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "False Positive Rate": fpr,
                "True Negatives": tn,
                "False Positives": fp,
                "False Negatives": fn,
                "True Positives": tp,
            }
        )

    results_df = pd.DataFrame(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    # Balanced operating point: maximize F1 on validation data.
    best_row = results_df.loc[results_df["F1 Score"].idxmax()]
    selected_threshold = float(best_row["Threshold"])

    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLD_PATH.write_text(
        f"{selected_threshold:.2f}\n",
        encoding="utf-8",
    )

    print("\n=== Random Forest Threshold Analysis ===")
    print(results_df.round(4).to_string(index=False))

    print("\n=== Selected Validation Threshold ===")
    print(f"Threshold: {selected_threshold:.2f}")
    print(f"Precision: {best_row['Precision']:.4f}")
    print(f"Recall: {best_row['Recall']:.4f}")
    print(f"F1 Score: {best_row['F1 Score']:.4f}")
    print(f"False Positive Rate: {best_row['False Positive Rate']:.4f}")

    print(f"\nResults saved to: {RESULTS_PATH}")
    print(f"Threshold saved to: {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()
