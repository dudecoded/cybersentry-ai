from pathlib import Path

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT_DIR / "models" / "random_forest.joblib"

THRESHOLD = 0.50

EXPECTED_FEATURES = [
    "dur", "proto", "service", "state", "spkts", "dpkts",
    "sbytes", "dbytes", "rate", "sttl", "dttl", "sload",
    "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit",
    "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
    "synack", "ackdat", "smean", "dmean", "trans_depth",
    "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports",
]

NUMERIC_FEATURES = [
    feature
    for feature in EXPECTED_FEATURES
    if feature not in {"proto", "service", "state"}
]

CATEGORICAL_FEATURES = ["proto", "service", "state"]

model = joblib.load(MODEL_PATH)


def predict_attack(features: dict) -> dict:
    if not isinstance(features, dict):
        raise ValueError("Features must be a JSON object.")

    missing = set(EXPECTED_FEATURES) - set(features)
    unexpected = set(features) - set(EXPECTED_FEATURES)

    if missing:
        raise ValueError(
            f"Missing required features: {sorted(missing)}"
        )

    if unexpected:
        raise ValueError(
            f"Unexpected features: {sorted(unexpected)}"
        )

    for feature in NUMERIC_FEATURES:
        value = features[feature]

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Feature '{feature}' must be numeric."
            )

        if not pd.notna(value) or abs(value) == float("inf"):
            raise ValueError(
                f"Feature '{feature}' must be finite."
            )

    for feature in CATEGORICAL_FEATURES:
        value = features[feature]

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Feature '{feature}' must be a non-empty string."
            )

    input_df = pd.DataFrame(
        [{feature: features[feature] for feature in EXPECTED_FEATURES}]
    )

    probability = float(model.predict_proba(input_df)[0][1])
    prediction = int(probability >= THRESHOLD)

    if probability >= 0.80:
        severity = "high"
    elif prediction == 1:
        severity = "medium"
    else:
        severity = "low"

    return {
        "prediction": prediction,
        "result": "Attack" if prediction == 1 else "Normal",
        "attack_probability": round(probability, 4),
        "threshold": THRESHOLD,
        "severity": severity,
    }