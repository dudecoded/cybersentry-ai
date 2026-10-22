from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_PATH = ROOT_DIR / "dataset" / "UNSW_NB15_testing-set.csv"
MODEL_DIR = ROOT_DIR / "models"

MODELS = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}


def main():
    print("Loading official test dataset...")

    test_df = pd.read_csv(TEST_PATH)
    X_test = test_df.drop(columns=["id", "attack_cat", "label"])
    y_test = test_df["label"]

    results = []

    for name, model_path in MODELS.items():
        print(f"\nEvaluating {name}...")

        model = joblib.load(model_path)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1-score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
            "False Positives": fp,
            "False Negatives": fn,
        })

        print(f"True Negatives: {tn:,}")
        print(f"False Positives: {fp:,}")
        print(f"False Negatives: {fn:,}")
        print(f"True Positives: {tp:,}")

    results_df = pd.DataFrame(results).set_index("Model")

    print("\n" + "=" * 90)
    print("CYBERSENTRY AI - OFFICIAL TEST COMPARISON")
    print("=" * 90)
    print(results_df.round(4).to_string())


if __name__ == "__main__":
    main()