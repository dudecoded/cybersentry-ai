from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT_DIR / "dataset" / "UNSW_NB15_training-set.csv"

MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"


def main():
    # 1. Load training dataset
    print("Loading training dataset...")

    train_df = pd.read_csv(TRAIN_PATH)

    # 2. Separate features and target
    X_train = train_df.drop(
        columns=["id", "attack_cat", "label"]
    )

    y_train = train_df["label"]

    print("Training shape:", X_train.shape)

    print("\nTarget distribution:")
    print(y_train.value_counts())

    # 3. Build preprocessing + classification pipeline
    print("\nBuilding pipeline...")

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

    # 4. Train the model
    print("\nTraining Logistic Regression...")

    model.fit(X_train, y_train)

    print("Training completed!")

    # 5. Save the complete pipeline
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()