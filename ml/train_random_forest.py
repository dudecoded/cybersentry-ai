from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT_DIR / "dataset" / "UNSW_NB15_training-set.csv"

MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "random_forest.joblib"


def main():
    print("Loading training dataset...")

    train_df = pd.read_csv(TRAIN_PATH)

    X_train = train_df.drop(columns=["id", "attack_cat", "label"])
    y_train = train_df["label"]

    print("Training shape:", X_train.shape)
    print("Building Random Forest pipeline...")

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

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Training completed!")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()