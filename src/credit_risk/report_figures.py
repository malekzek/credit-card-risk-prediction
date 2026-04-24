import json
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from .config import (
    DATA_QUALITY_FIG_PATH,
    MODEL_COMPARISON_FIG_PATH,
    MODEL_COMPARISON_JSON_PATH,
    PIPELINE_OVERVIEW_FIG_PATH,
    REPORT_DIR,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _require_file(file_path: Path, run_hint: str) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}. {run_hint}")


def _build_pipeline_figure(output_path: Path) -> None:
    steps = [
        "Raw Data\n(Home Credit CSVs)",
        "Data Quality +\nFeature Engineering",
        "Model Training +\nCross Validation",
        "Explainability\n(SHAP)",
        "FastAPI Inference +\nDocker",
    ]

    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_positions = [0.02, 0.22, 0.42, 0.62, 0.82]
    box_width = 0.16
    box_height = 0.44

    for idx, step_label in enumerate(steps):
        x = x_positions[idx]
        box = FancyBboxPatch(
            (x, 0.28),
            box_width,
            box_height,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.5,
            edgecolor="#1d3557",
            facecolor="#f1f5fb",
        )
        ax.add_patch(box)
        ax.text(x + box_width / 2, 0.5, step_label, ha="center", va="center", fontsize=10)

        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - 0.01, 0.5),
                xytext=(x + box_width + 0.01, 0.5),
                arrowprops={"arrowstyle": "->", "linewidth": 1.7, "color": "#457b9d"},
            )

    ax.text(
        0.5,
        0.9,
        "Credit Risk Prediction Pipeline",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#1d3557",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_model_comparison_figure(model_comparison_path: Path, output_path: Path) -> None:
    comparison_data = json.loads(model_comparison_path.read_text(encoding="utf-8"))
    valid_models = [
        model_row
        for model_row in comparison_data.get("models", [])
        if model_row.get("status") == "ok" and model_row.get("cv_auc_mean") is not None
    ]
    if not valid_models:
        raise ValueError("No successful models with cv_auc_mean found in model comparison report.")

    score_df = pd.DataFrame(valid_models)[["model", "cv_auc_mean"]].sort_values(
        "cv_auc_mean", ascending=False
    )
    best_model = score_df.iloc[0]["model"]
    colors = ["#2a9d8f" if model_name == best_model else "#9ecae1" for model_name in score_df["model"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(score_df["model"], score_df["cv_auc_mean"], color=colors, edgecolor="#264653")
    ax.set_title("Model Comparison (3-Fold CV ROC-AUC)")
    ax.set_xlabel("Model")
    ax.set_ylabel("CV ROC-AUC")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    min_score = float(score_df["cv_auc_mean"].min())
    max_score = float(score_df["cv_auc_mean"].max())
    ax.set_ylim(max(0.5, min_score - 0.01), min(1.0, max_score + 0.01))

    for bar, score in zip(bars, score_df["cv_auc_mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(score) + 0.0004,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_data_quality_figure(data_quality_json_path: Path, missingness_csv_path: Path, output_path: Path) -> None:
    quality_report = json.loads(data_quality_json_path.read_text(encoding="utf-8"))
    target_info = quality_report.get("application_train_target", {})

    target_counts = [
        int(target_info.get("target_0_count", 0)),
        int(target_info.get("target_1_count", 0)),
    ]
    target_ratios = [
        float(target_info.get("target_0_ratio", 0.0)),
        float(target_info.get("target_1_ratio", 0.0)),
    ]

    missingness_df = pd.read_csv(missingness_csv_path)
    top_missingness_df = missingness_df.head(15).copy().sort_values("missing_percent", ascending=True)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))

    labels = ["TARGET=0\n(No default)", "TARGET=1\n(Default)"]
    bars = ax_left.bar(labels, target_counts, color=["#90be6d", "#f94144"], edgecolor="#264653")
    ax_left.set_title("Target Distribution")
    ax_left.set_ylabel("Row count")
    for bar, ratio in zip(bars, target_ratios):
        ax_left.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{ratio:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax_right.barh(
        top_missingness_df["column"],
        top_missingness_df["missing_percent"],
        color="#577590",
        edgecolor="#264653",
    )
    ax_right.set_title("Top 15 Missing Columns")
    ax_right.set_xlabel("Missing percent")
    ax_right.set_ylabel("Column")

    fig.suptitle("Data Quality Overview", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_project_figures() -> dict:
    """Generate presentation-ready matplotlib figures from existing report artifacts."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_comparison_path = MODEL_COMPARISON_JSON_PATH
    data_quality_json_path = REPORT_DIR / "data_quality_report.json"
    missingness_csv_path = REPORT_DIR / "application_train_missingness.csv"

    _require_file(model_comparison_path, "Run: python compare_models.py")
    _require_file(data_quality_json_path, "Run: python generate_data_report.py")
    _require_file(missingness_csv_path, "Run: python generate_data_report.py")

    _build_pipeline_figure(PIPELINE_OVERVIEW_FIG_PATH)
    _build_model_comparison_figure(model_comparison_path, MODEL_COMPARISON_FIG_PATH)
    _build_data_quality_figure(data_quality_json_path, missingness_csv_path, DATA_QUALITY_FIG_PATH)

    summary = {
        "pipeline_overview": str(PIPELINE_OVERVIEW_FIG_PATH),
        "model_comparison": str(MODEL_COMPARISON_FIG_PATH),
        "data_quality_overview": str(DATA_QUALITY_FIG_PATH),
    }

    print(f"Saved pipeline overview figure: {PIPELINE_OVERVIEW_FIG_PATH}")
    print(f"Saved model comparison figure: {MODEL_COMPARISON_FIG_PATH}")
    print(f"Saved data quality figure: {DATA_QUALITY_FIG_PATH}")

    return summary


if __name__ == "__main__":
    generate_project_figures()
