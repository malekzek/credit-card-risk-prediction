import json
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from .config import (
    MODEL_COMPARISON_CSV_PATH,
    MODEL_COMPARISON_JSON_PATH,
    RANDOM_STATE,
    REPORT_DIR,
)
from .data import load_train_features
from .features import get_feature_types, split_features_target

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_IMPORT_ERROR = ""
except Exception as exc:
    LGBMClassifier = None
    LIGHTGBM_IMPORT_ERROR = str(exc)

try:
    from catboost import CatBoostClassifier

    CATBOOST_IMPORT_ERROR = ""
except Exception as exc:
    CatBoostClassifier = None
    CATBOOST_IMPORT_ERROR = str(exc)


MAX_COMPARE_ROWS = 40000
CV_FOLDS = 3


def _build_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
        ]
    )


def _hgb_factory() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.05,
        max_iter=120,
        max_bins=64,
        random_state=RANDOM_STATE,
    )


def _lgbm_factory():
    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def _catboost_factory():
    return CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=False,
    )


def _model_registry() -> list[dict]:
    registry = [
        {
            "name": "hist_gradient_boosting",
            "factory": _hgb_factory,
            "available": True,
            "reason": "",
        }
    ]

    registry.append(
        {
            "name": "lightgbm",
            "factory": _lgbm_factory if LGBMClassifier is not None else None,
            "available": LGBMClassifier is not None,
            "reason": LIGHTGBM_IMPORT_ERROR,
        }
    )

    registry.append(
        {
            "name": "catboost",
            "factory": _catboost_factory if CatBoostClassifier is not None else None,
            "available": CatBoostClassifier is not None,
            "reason": CATBOOST_IMPORT_ERROR,
        }
    )

    return registry


def run_model_comparison() -> dict:
    """Run stratified CV comparison for baseline and gradient boosting models."""
    df_train = load_train_features(use_secondary_features=True)
    print(f"Model comparison train table shape: {df_train.shape}")

    X, y = split_features_target(df_train)
    if len(X) > MAX_COMPARE_ROWS:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=MAX_COMPARE_ROWS,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        print(f"Using stratified sample of {MAX_COMPARE_ROWS} rows for model comparison.")

    numeric_cols, categorical_cols = get_feature_types(X)
    if not numeric_cols:
        raise ValueError("No numeric columns found for model comparison.")

    if categorical_cols:
        print(f"Info: ignoring {len(categorical_cols)} categorical columns in comparison.")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    registry = _model_registry()

    results: list[dict] = []
    for model_spec in registry:
        name = model_spec["name"]
        if not model_spec["available"]:
            reason = model_spec["reason"] or "package not available"
            print(f"Skipping {name}: {reason}")
            results.append(
                {
                    "model": name,
                    "status": "skipped",
                    "reason": reason,
                    "cv_auc_mean": None,
                    "cv_auc_std": None,
                    "cv_auc_scores": [],
                }
            )
            continue

        print(f"Evaluating {name}...")
        fold_scores: list[float] = []
        factory: Callable = model_spec["factory"]

        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_valid = X.iloc[valid_idx]
            y_fold_valid = y.iloc[valid_idx]

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", _build_preprocessor(numeric_cols)),
                    ("model", factory()),
                ]
            )
            pipeline.fit(X_fold_train, y_fold_train)

            fold_pred = pipeline.predict_proba(X_fold_valid)[:, 1]
            fold_auc = float(roc_auc_score(y_fold_valid, fold_pred))
            fold_scores.append(fold_auc)
            print(f"  Fold {fold_idx}/{CV_FOLDS} ROC-AUC: {fold_auc:.5f}")

        mean_auc = float(np.mean(fold_scores))
        std_auc = float(np.std(fold_scores))
        print(f"  {name} mean ROC-AUC: {mean_auc:.5f}")
        print(f"  {name} std ROC-AUC: {std_auc:.5f}")

        results.append(
            {
                "model": name,
                "status": "ok",
                "reason": "",
                "cv_auc_mean": mean_auc,
                "cv_auc_std": std_auc,
                "cv_auc_scores": fold_scores,
            }
        )

    valid_results = [row for row in results if row["status"] == "ok"]
    ranking = sorted(valid_results, key=lambda x: x["cv_auc_mean"], reverse=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    comparison_report = {
        "cv_folds": CV_FOLDS,
        "rows_used": int(len(X)),
        "models": results,
        "ranking": [row["model"] for row in ranking],
    }

    MODEL_COMPARISON_JSON_PATH.write_text(
        json.dumps(comparison_report, indent=2),
        encoding="utf-8",
    )

    flat_rows = []
    for row in results:
        flat_rows.append(
            {
                "model": row["model"],
                "status": row["status"],
                "cv_auc_mean": row["cv_auc_mean"],
                "cv_auc_std": row["cv_auc_std"],
                "reason": row["reason"],
            }
        )

    comparison_df = pd.DataFrame(flat_rows)
    comparison_df.to_csv(MODEL_COMPARISON_CSV_PATH, index=False)

    print(f"Saved model comparison JSON: {MODEL_COMPARISON_JSON_PATH}")
    print(f"Saved model comparison CSV: {MODEL_COMPARISON_CSV_PATH}")

    if ranking:
        print("Model ranking (best to worst):")
        for idx, row in enumerate(ranking, start=1):
            print(f"  {idx}. {row['model']} ({row['cv_auc_mean']:.5f})")

    return comparison_report


if __name__ == "__main__":
    run_model_comparison()
