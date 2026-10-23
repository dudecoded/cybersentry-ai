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

    train_conflicts = (
        train.groupby(FEATURE_COLUMNS, dropna=False)["label"].nunique() > 1
    )

    test_conflicts = (
        test.groupby(FEATURE_COLUMNS, dropna=False)["label"].nunique() > 1
    )

    train_conflict_features = train_conflicts[train_conflicts].index
    test_conflict_features = test_conflicts[test_conflicts].index

    train_conflict_df = train.set_index(FEATURE_COLUMNS)
    test_conflict_df = test.set_index(FEATURE_COLUMNS)

    train_ambiguous = train_conflict_df.loc[
        train_conflict_features
    ].reset_index()

    test_ambiguous = test_conflict_df.loc[
        test_conflict_features
    ].reset_index()

    print("\n" + "=" * 60)
    print("DISTINCT CONFLICTING FEATURE GROUPS")
    print("=" * 60)

    print("Training groups:", len(train_conflict_features))
    print("Testing groups:", len(test_conflict_features))

    train_keys = set(train_conflict_features)
    test_keys = set(test_conflict_features)

    print("Conflicting groups in both datasets:", len(train_keys & test_keys))

    train_labels = train_ambiguous.groupby(FEATURE_COLUMNS)["label"].agg(
        lambda values: sorted(set(values))
    )

    test_labels = test_ambiguous.groupby(FEATURE_COLUMNS)["label"].agg(
        lambda values: sorted(set(values))
    )

    print("\nTraining conflicting label combinations:")
    print(train_labels.value_counts().to_string())

    print("\nTesting conflicting label combinations:")
    print(test_labels.value_counts().to_string())

    test_features_in_train = test[FEATURE_COLUMNS].merge(
        train[FEATURE_COLUMNS].drop_duplicates(),
        on=FEATURE_COLUMNS,
        how="inner",
    )

    print("\nTest feature rows matching training features:",
          len(test_features_in_train))


if __name__ == "__main__":
    main()