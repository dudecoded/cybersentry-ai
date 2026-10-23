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

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train_groups = train.groupby(FEATURE_COLUMNS)["label"].agg(
        lambda labels: frozenset(labels)
    )

    test_groups = test.groupby(FEATURE_COLUMNS)["label"].agg(
        lambda labels: frozenset(labels)
    )

    common_features = train_groups.index.intersection(test_groups.index)

    conflicting_features = [
        features
        for features in common_features
        if train_groups.loc[features].isdisjoint(test_groups.loc[features])
    ]

    test_index = pd.MultiIndex.from_frame(test[FEATURE_COLUMNS])
    affected_mask = test_index.isin(conflicting_features)

    affected = test.loc[affected_mask]

    print("\n" + "=" * 60)
    print("CONFLICTING RECORD IMPACT")
    print("=" * 60)

    print("Distinct conflicting feature groups:", len(conflicting_features))
    print("Affected test records:", len(affected))
    print(
        "Percentage of test dataset affected:",
        f"{len(affected) / len(test) * 100:.4f}%",
    )

    print("\nAffected test record labels:")

    for label, count in affected["label"].value_counts().sort_index().items():
        name = "Normal" if label == 0 else "Attack"
        print(f"{name} ({label}): {count:,}")

    print("\nAffected records by label percentage:")

    if not affected.empty:
        print(
            (affected["label"].value_counts(normalize=True) * 100)
            .sort_index()
            .round(2)
            .to_string()
        )

    print("\nDiagnostic completed.")


if __name__ == "__main__":
    main()