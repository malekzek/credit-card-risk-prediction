import json
from pathlib import Path

import pandas as pd

from .config import REPORT_DIR, RAW_DATA_DIR, TRAIN_FILE


PROFILE_SAMPLE_ROWS = 10000


def _read_csv_safely(csv_path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with encoding fallbacks for mixed-source files."""
    try:
        return pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace", **kwargs)
    except Exception:
        for enc in ("latin1", "cp1252"):
            try:
                return pd.read_csv(csv_path, encoding=enc, **kwargs)
            except Exception:
                continue
        raise


def _summarize_single_file(csv_path: Path) -> dict:
    """Return fast sampled quality stats for one CSV file."""
    df_sample = _read_csv_safely(csv_path, nrows=PROFILE_SAMPLE_ROWS)

    rows, cols = df_sample.shape
    missing_cells = int(df_sample.isna().sum().sum())
    total_cells = rows * cols if rows and cols else 0
    missing_ratio = float(missing_cells / total_cells) if total_cells else 0.0
    file_size_mb = float(csv_path.stat().st_size / (1024 * 1024))

    return {
        "file_name": csv_path.name,
        "rows_profiled": int(rows),
        "columns": int(cols),
        "missing_cells_in_sample": missing_cells,
        "missing_ratio_in_sample": missing_ratio,
        "profile_sample_rows": PROFILE_SAMPLE_ROWS,
        "file_size_mb": file_size_mb,
    }


def _application_train_details() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Compute target distribution, missingness by column, and dtype summary."""
    train_df = _read_csv_safely(TRAIN_FILE)

    if "TARGET" not in train_df.columns:
        raise ValueError("TARGET column is missing in application_train.csv")

    target_counts = train_df["TARGET"].value_counts(dropna=False).sort_index()
    target_ratios = train_df["TARGET"].value_counts(normalize=True, dropna=False).sort_index()

    target_summary = {
        "target_0_count": int(target_counts.get(0, 0)),
        "target_1_count": int(target_counts.get(1, 0)),
        "target_0_ratio": float(target_ratios.get(0, 0.0)),
        "target_1_ratio": float(target_ratios.get(1, 0.0)),
    }

    missingness = (
        train_df.isna()
        .mean()
        .mul(100)
        .rename("missing_percent")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("missing_percent", ascending=False)
    )

    dtypes = (
        train_df.dtypes.astype(str)
        .rename("dtype")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("column")
    )

    return target_summary, missingness, dtypes


def generate_data_quality_report() -> dict:
    """Generate and save quality report files in reports/."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")

    file_summaries = [_summarize_single_file(path) for path in csv_files]
    target_summary, missingness_df, dtypes_df = _application_train_details()

    report = {
        "raw_data_dir": str(RAW_DATA_DIR),
        "num_files": len(file_summaries),
        "files": file_summaries,
        "application_train_target": target_summary,
        "notes": "Baseline quality summary before multi-table feature engineering.",
    }

    report_path = REPORT_DIR / "data_quality_report.json"
    missingness_path = REPORT_DIR / "application_train_missingness.csv"
    dtypes_path = REPORT_DIR / "application_train_dtypes.csv"

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    missingness_df.to_csv(missingness_path, index=False)
    dtypes_df.to_csv(dtypes_path, index=False)

    print(f"Saved report JSON: {report_path}")
    print(f"Saved missingness CSV: {missingness_path}")
    print(f"Saved dtypes CSV: {dtypes_path}")

    return report


if __name__ == "__main__":
    generate_data_quality_report()
