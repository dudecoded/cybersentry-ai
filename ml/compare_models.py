from pathlib import Path

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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

    models = {
        "Dummy Classifier": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    for name, classifier in models.items():
        print(f"\nTraining {name}...")

        model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        probabilities = model.predict_proba(X_val)[:, 1]

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_val, predictions),
                "Precision": precision_score(
                    y_val, predictions, zero_division=0
                ),
                "Recall": recall_score(
                    y_val, predictions, zero_division=0
                ),
                "F1-score": f1_score(
                    y_val, predictions, zero_division=0
                ),
                "ROC-AUC": roc_auc_score(y_val, probabilities),
            }
        )

        print(f"Completed {name}.")

    results_df = pd.DataFrame(results).set_index("Model")

    print("\n" + "=" * 75)
    print("CYBERSENTRY AI - VALIDATION MODEL COMPARISON")
    print("=" * 75)
    print(results_df.round(4).to_string())


if __name__ == "__main__":
    main()