import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from .config import METRICS_PATH, MODEL_DIR, MODEL_PATH, RANDOM_STATE, REPORT_DIR
from .data import load_train_features
from .features import get_feature_types, split_features_target


MAX_TRAIN_ROWS = 120000
MAX_CV_ROWS = 90000
CV_FOLDS = 3


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    """Create a fast baseline pipeline for large tabular data."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
        ]
    )

    model = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.05,
        max_iter=150,
        max_bins=64,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def run_stratified_cv(
    X: "pd.DataFrame",
    y: "pd.Series",
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict:
    """Run stratified K-fold CV and return fold scores."""
    X_cv, y_cv = X, y

    if len(X_cv) > MAX_CV_ROWS:
        X_cv, _, y_cv, _ = train_test_split(
            X_cv,
            y_cv,
            train_size=MAX_CV_ROWS,
            random_state=RANDOM_STATE,
            stratify=y_cv,
        )
        print(f"Using stratified sample of {MAX_CV_ROWS} rows for cross-validation.")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores: list[float] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X_cv, y_cv), start=1):
        X_fold_train = X_cv.iloc[train_idx]
        y_fold_train = y_cv.iloc[train_idx]
        X_fold_valid = X_cv.iloc[valid_idx]
        y_fold_valid = y_cv.iloc[valid_idx]

        fold_pipeline = build_pipeline(numeric_cols, categorical_cols)
        fold_pipeline.fit(X_fold_train, y_fold_train)

        fold_pred = fold_pipeline.predict_proba(X_fold_valid)[:, 1]
        fold_auc = roc_auc_score(y_fold_valid, fold_pred)
        fold_scores.append(float(fold_auc))
        print(f"Fold {fold_idx}/{CV_FOLDS} ROC-AUC: {fold_auc:.5f}")

    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))

    print(f"CV ROC-AUC mean: {mean_auc:.5f}")
    print(f"CV ROC-AUC std: {std_auc:.5f}")

    return {
        "scores": fold_scores,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "rows_used": int(len(X_cv)),
    }


def run_training() -> None:
    """Train a model with stratified CV and save model + metrics."""
    df_train = load_train_features(use_secondary_features=True)
    print(f"Training table shape: {df_train.shape}")

    X, y = split_features_target(df_train)

    if len(X) > MAX_TRAIN_ROWS:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=MAX_TRAIN_ROWS,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        print(f"Using stratified sample of {MAX_TRAIN_ROWS} rows for faster training.")

    numeric_cols, categorical_cols = get_feature_types(X)
    if not numeric_cols:
        raise ValueError("No numeric columns found in training data.")

    if categorical_cols:
        print(f"Info: ignoring {len(categorical_cols)} categorical columns in baseline model.")

    cv_result = run_stratified_cv(X, y, numeric_cols, categorical_cols)

    pipeline = build_pipeline(numeric_cols, categorical_cols)

    pipeline.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    metrics = {
        "validation_roc_auc": float(cv_result["mean_auc"]),
        "cv_auc_scores": cv_result["scores"],
        "cv_auc_mean": float(cv_result["mean_auc"]),
        "cv_auc_std": float(cv_result["std_auc"]),
        "cv_folds": CV_FOLDS,
        "cv_rows_used": int(cv_result["rows_used"]),
        "train_rows_used": int(len(X)),
        "model": "hist_gradient_boosting_with_secondary_aggregates",
        "notes": "Uses application table + applicant-level aggregated secondary features with stratified cross-validation.",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    run_training()
