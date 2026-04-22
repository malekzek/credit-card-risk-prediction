import argparse
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from credit_risk.explainability import run_explainability_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP explainability report.")
    parser.add_argument(
        "--model",
        default="catboost",
        help=(
            "Explainability backend to run. "
            "Default is catboost. You can also use auto (from model_comparison.csv) or lightgbm."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_explainability_report(model_name=args.model)
