import pandas as pd

from .config import (
    PROCESSED_DATA_DIR,
    TEST_FEATURES_FILE,
    TEST_FILE,
    TRAIN_FEATURES_FILE,
    TRAIN_FILE,
)
from .secondary_features import load_or_build_secondary_features


def load_application_train() -> pd.DataFrame:
    """Load the main training table."""
    return pd.read_csv(TRAIN_FILE)


def load_application_test() -> pd.DataFrame:
    """Load the main test table."""
    return pd.read_csv(TEST_FILE)


def build_processed_feature_tables(force_rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and cache train/test feature tables with secondary aggregations."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TRAIN_FEATURES_FILE.exists() and TEST_FEATURES_FILE.exists() and not force_rebuild:
        print(f"Loading cached train features: {TRAIN_FEATURES_FILE}")
        print(f"Loading cached test features: {TEST_FEATURES_FILE}")
        train_features = pd.read_pickle(TRAIN_FEATURES_FILE)
        test_features = pd.read_pickle(TEST_FEATURES_FILE)
        return train_features, test_features

    train_df = load_application_train()
    test_df = load_application_test()
    secondary_features = load_or_build_secondary_features(force_rebuild=force_rebuild)

    train_features = train_df.merge(secondary_features, on="SK_ID_CURR", how="left")
    test_features = test_df.merge(secondary_features, on="SK_ID_CURR", how="left")

    train_features.to_pickle(TRAIN_FEATURES_FILE)
    test_features.to_pickle(TEST_FEATURES_FILE)

    print(f"Saved train features: {TRAIN_FEATURES_FILE}")
    print(f"Saved test features: {TEST_FEATURES_FILE}")

    return train_features, test_features


def load_train_features(use_secondary_features: bool = True) -> pd.DataFrame:
    """Load train data with optional secondary-table aggregated features."""
    if not use_secondary_features:
        return load_application_train()

    train_features, _ = build_processed_feature_tables(force_rebuild=False)
    return train_features


def load_test_features(use_secondary_features: bool = True) -> pd.DataFrame:
    """Load test data with optional secondary-table aggregated features."""
    if not use_secondary_features:
        return load_application_test()

    _, test_features = build_processed_feature_tables(force_rebuild=False)
    return test_features
