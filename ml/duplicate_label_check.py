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

    train_features = train_df[FEATURE_COLUMNS]
    test_features = test_df[FEATURE_COLUMNS]

    train_check = train_features.copy()
    train_check["label"] = train_df["label"]

    test_check = test_features.copy()
    test_check["label"] = test_df["label"]

    train_unique = train_check.drop_duplicates()

    test_matches = test_check.merge(
        train_unique,
        on=FEATURE_COLUMNS,
        how="inner",
        suffixes=("_test", "_train"),
    )

    print("\n" + "=" * 60)
    print("DUPLICATE LABEL CONSISTENCY")
    print("=" * 60)

    print("Matching feature-label combinations:", len(test_matches))

    if not test_matches.empty:
        conflicting = (
            test_matches["label_test"] != test_matches["label_train"]
        )

        print("Matching records with conflicting labels:", int(conflicting.sum()))

    train_label_counts = train_check.groupby(FEATURE_COLUMNS)["label"].nunique()

    test_label_counts = test_check.groupby(FEATURE_COLUMNS)["label"].nunique()

    print(
        "Training feature groups with conflicting labels:",
        int((train_label_counts > 1).sum()),
    )

    print(
        "Testing feature groups with conflicting labels:",
        int((test_label_counts > 1).sum()),
    )


if __name__ == "__main__":
    main()