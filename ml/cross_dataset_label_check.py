from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT_DIR / "dataset" / "UNSW_NB15_training-set.csv"
TEST_PATH = ROOT_DIR / "dataset" / "UNSW_NB15_testing-set.csv"

FEATURE_COLUMNS = [
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


def main():
    print("Loading datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train = train_df[FEATURE_COLUMNS + ["label"]].drop_duplicates()
    test = test_df[FEATURE_COLUMNS + ["label"]].drop_duplicates()

    train_groups = train.groupby(FEATURE_COLUMNS)["label"].agg(
        lambda labels: frozenset(labels)
    )

    test_groups = test.groupby(FEATURE_COLUMNS)["label"].agg(
        lambda labels: frozenset(labels)
    )

    common_features = train_groups.index.intersection(test_groups.index)

    results = []

    for features in common_features:
        train_labels = train_groups.loc[features]
        test_labels = test_groups.loc[features]

        results.append({
    "train_labels": train_labels,
    "test_labels": test_labels,
    "same_label_exists": bool(train_labels & test_labels),
    "conflicting_label_exists": not bool(train_labels & test_labels),
})

    comparison = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("CROSS-DATASET LABEL COMPARISON")
    print("=" * 60)

    print("Distinct feature groups shared:", len(comparison))
    print(
        "Groups sharing at least one label:",
        int(comparison["same_label_exists"].sum()),
    )
    print(
        "Groups with different label sets:",
        int(comparison["conflicting_label_exists"].sum()),
    )

    print("\nTraining label combinations:")
    print(comparison["train_labels"].value_counts().to_string())

    print("\nTesting label combinations:")
    print(comparison["test_labels"].value_counts().to_string())


if __name__ == "__main__":
    main()