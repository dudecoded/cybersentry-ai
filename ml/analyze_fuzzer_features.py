from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT / "dataset" / "UNSW_NB15_testing-set.csv"
MODEL_PATH = ROOT / "models" / "random_forest.joblib"

df = pd.read_csv(TEST_PATH)

X = df.drop(columns=["id", "attack_cat", "label"])
y = df["label"]

model = joblib.load(MODEL_PATH)

# Process in batches to reduce memory usage.
probabilities = []

for start in range(0, len(df), 2000):
    batch = X.iloc[start:start + 2000]
    probabilities.extend(model.predict_proba(batch)[:, 1])

df["attack_probability"] = probabilities

print("\n=== Attack Probability by Category ===")

summary = (
    df.groupby("attack_cat")["attack_probability"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(4)
)

print(summary.to_string())

print("\n=== Fuzzer vs Normal ===")

for category in ["Normal", "Fuzzers"]:
    values = df.loc[
        df["attack_cat"] == category,
        "attack_probability"
    ]

    detected = (values >= 0.50).sum()

    print(
        f"{category}: "
        f"{detected}/{len(values)} classified as attacks "
        f"({detected / len(values) * 100:.2f}%)"
    )