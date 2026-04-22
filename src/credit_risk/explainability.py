import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    EXPLAINABILITY_JSON_PATH,
    EXPLAINABILITY_PLOT_PATH,
    EXPLAINABILITY_TOP_FEATURES_CSV_PATH,
    MODEL_COMPARISON_CSV_PATH,
    RANDOM_STATE,
    REPORT_DIR,
)
from .data import load_train_features
from .features import get_feature_types, split_features_target

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from catboost import CatBoostClassifier, Pool

    CATBOOST_IMPORT_ERROR = ""
except Exception as exc:
    CatBoostClassifier = None
    Pool = None
    CATBOOST_IMPORT_ERROR = str(exc)

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_IMPORT_ERROR = ""
except Exception as exc:
    LGBMClassifier = None
    LIGHTGBM_IMPORT_ERROR = str(exc)

MAX_TRAIN_ROWS_FOR_EXPLAINABILITY = 70000
MAX_EXPLAIN_ROWS = 5000
TOP_FEATURES_TO_SAVE = 100
TOP_FEATURES_TO_PLOT = 25
SUPPORTED_EXPLAINABILITY_MODELS = {"catboost", "lightgbm"}
MODEL_ALIASES = {
    "lgbm": "lightgbm",
    "hgb": "hist_gradient_boosting",
}


def _read_model_comparison_winner() -> str:
    if not MODEL_COMPARISON_CSV_PATH.exists():
        return "catboost"

    comparison_df = pd.read_csv(MODEL_COMPARISON_CSV_PATH)
    if "status" not in comparison_df.columns or "cv_auc_mean" not in comparison_df.columns:
        return "catboost"

    valid_df = comparison_df[comparison_df["status"] == "ok"].copy()
    valid_df = valid_df.dropna(subset=["cv_auc_mean"])
    if valid_df.empty:
        return "catboost"

    valid_df = valid_df.sort_values("cv_auc_mean", ascending=False)
    return str(valid_df.iloc[0]["model"])


def _normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower()
    return MODEL_ALIASES.get(normalized, normalized)


def _resolve_requested_model(model_name: str) -> str:
    if model_name != "auto":
        return _normalize_model_name(model_name)
    return _normalize_model_name(_read_model_comparison_winner())


def _prepare_features_for_catboost(X: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    X_prepared = X.copy()
    for col in categorical_cols:
        X_prepared[col] = X_prepared[col].fillna("__MISSING__").astype(str)
    return X_prepared


def _prepare_features_for_lightgbm(X: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    X_prepared = X.copy()
    for col in categorical_cols:
        X_prepared[col] = X_prepared[col].fillna("__MISSING__").astype("category")
    return X_prepared


def _sample_rows(X: pd.DataFrame, y: pd.Series, max_rows: int) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= max_rows:
        return X, y

    X_sampled, _, y_sampled, _ = train_test_split(
        X,
        y,
        train_size=max_rows,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_sampled, y_sampled


def _build_top_features_plot(importance_df: pd.DataFrame, output_path: Path, model_label: str) -> None:
    pretty_model_name = {
        "catboost": "CatBoost",
        "lightgbm": "LightGBM",
    }.get(model_label, model_label)

    top_plot_df = importance_df.head(TOP_FEATURES_TO_PLOT).copy()
    top_plot_df = top_plot_df.sort_values("mean_abs_shap", ascending=True)

    height = max(6, 0.28 * len(top_plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(11, height))
    ax.barh(top_plot_df["feature"], top_plot_df["mean_abs_shap"], color="#1d6fa5")
    ax.set_title(f"SHAP Feature Impact ({pretty_model_name})")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _compute_catboost_shap(
    X: pd.DataFrame,
    y: pd.Series,
    X_explain: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    if CatBoostClassifier is None or Pool is None:
        raise ImportError(f"catboost is required for explainability: {CATBOOST_IMPORT_ERROR}")

    model = CatBoostClassifier(
        iterations=250,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=False,
    )
    model.fit(X, y, cat_features=categorical_cols)

    explain_pool = Pool(X_explain, cat_features=categorical_cols)
    shap_values = model.get_feature_importance(explain_pool, type="ShapValues")

    contribution_values = shap_values[:, :-1]
    expected_values = shap_values[:, -1]
    return contribution_values, expected_values


def _compute_lightgbm_shap(
    X: pd.DataFrame,
    y: pd.Series,
    X_explain: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    if LGBMClassifier is None:
        raise ImportError(f"lightgbm is required for explainability: {LIGHTGBM_IMPORT_ERROR}")

    model = LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X, y, categorical_feature=categorical_cols if categorical_cols else "auto")

    shap_values = model.predict(X_explain, pred_contrib=True)
    if isinstance(shap_values, list):
        if not shap_values:
            raise ValueError("LightGBM returned empty SHAP contribution output.")
        shap_values = shap_values[-1]

    shap_values = np.asarray(shap_values)
    if shap_values.ndim != 2 or shap_values.shape[1] != X_explain.shape[1] + 1:
        raise ValueError(f"Unexpected LightGBM SHAP output shape: {shap_values.shape}")

    contribution_values = shap_values[:, :-1]
    expected_values = shap_values[:, -1]
    return contribution_values, expected_values


def run_explainability_report(model_name: str = "catboost") -> dict:
    """Train an explainability model and save SHAP feature impact artifacts."""
    requested_model = _resolve_requested_model(model_name)
    if model_name != "auto" and requested_model not in SUPPORTED_EXPLAINABILITY_MODELS:
        supported_values = ["auto", "catboost", "lightgbm"]
        raise ValueError(
            f"Unsupported model '{model_name}' for explainability. "
            f"Supported values: {', '.join(supported_values)}"
        )

    explainer_model = requested_model
    if requested_model not in SUPPORTED_EXPLAINABILITY_MODELS:
        print(
            "Info: winner model is "
            f"'{requested_model}', but SHAP report supports only catboost/lightgbm. "
            "Falling back to catboost."
        )
        explainer_model = "catboost"

    if explainer_model == "catboost" and (CatBoostClassifier is None or Pool is None):
        raise ImportError(f"catboost is required for explainability: {CATBOOST_IMPORT_ERROR}")

    if explainer_model == "lightgbm" and LGBMClassifier is None:
        raise ImportError(f"lightgbm is required for explainability: {LIGHTGBM_IMPORT_ERROR}")

    df_train = load_train_features(use_secondary_features=True)
    print(f"Explainability train table shape: {df_train.shape}")

    X, y = split_features_target(df_train)
    X, y = _sample_rows(X, y, MAX_TRAIN_ROWS_FOR_EXPLAINABILITY)
    if len(X) < len(df_train):
        print(f"Using stratified sample of {len(X)} rows for explainability training.")

    numeric_cols, categorical_cols = get_feature_types(X)
    if not numeric_cols:
        raise ValueError("No numeric columns found for explainability.")

    if categorical_cols:
        print(f"Using {len(numeric_cols)} numeric + {len(categorical_cols)} categorical columns.")
    else:
        print(f"Using {len(numeric_cols)} numeric columns.")

    if explainer_model == "catboost":
        X = _prepare_features_for_catboost(X, categorical_cols)
    else:
        X = _prepare_features_for_lightgbm(X, categorical_cols)

    explain_rows = min(MAX_EXPLAIN_ROWS, len(X))
    if explain_rows < len(X):
        _, X_explain, _, _ = train_test_split(
            X,
            y,
            test_size=explain_rows,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    else:
        X_explain = X

    if explainer_model == "catboost":
        print("Training CatBoost model for SHAP explainability...")
        contribution_values, expected_values = _compute_catboost_shap(X, y, X_explain, categorical_cols)
    else:
        print("Training LightGBM model for SHAP explainability...")
        contribution_values, expected_values = _compute_lightgbm_shap(X, y, X_explain, categorical_cols)

    importance_df = pd.DataFrame(
        {
            "feature": X_explain.columns,
            "mean_abs_shap": np.abs(contribution_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    importance_df["rank"] = np.arange(1, len(importance_df) + 1)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    EXPLAINABILITY_TOP_FEATURES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    importance_df.head(TOP_FEATURES_TO_SAVE).to_csv(EXPLAINABILITY_TOP_FEATURES_CSV_PATH, index=False)
    _build_top_features_plot(importance_df, EXPLAINABILITY_PLOT_PATH, explainer_model)

    summary = {
        "model_argument": model_name,
        "requested_model": requested_model,
        "explainer_model": explainer_model,
        "train_rows_used": int(len(X)),
        "rows_explained": int(len(X_explain)),
        "feature_count": int(X_explain.shape[1]),
        "top_features_file": str(EXPLAINABILITY_TOP_FEATURES_CSV_PATH),
        "summary_plot_file": str(EXPLAINABILITY_PLOT_PATH),
        "mean_expected_value": float(np.mean(expected_values)),
        "top_features": importance_df.head(10).to_dict(orient="records"),
    }

    EXPLAINABILITY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved explainability summary JSON: {EXPLAINABILITY_JSON_PATH}")
    print(f"Saved explainability top features CSV: {EXPLAINABILITY_TOP_FEATURES_CSV_PATH}")
    print(f"Saved explainability summary plot: {EXPLAINABILITY_PLOT_PATH}")

    return summary


if __name__ == "__main__":
    run_explainability_report()
