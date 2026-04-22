from typing import Tuple

import pandas as pd


def split_features_target(df: pd.DataFrame, target_col: str = "TARGET") -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def get_feature_types(X: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """Identify numeric and categorical column names."""
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols
