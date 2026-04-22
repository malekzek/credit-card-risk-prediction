from pathlib import Path
import argparse
import sys

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from credit_risk.data import build_processed_feature_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cached train/test feature tables with secondary aggregations.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild feature caches even if they already exist.",
    )
    args = parser.parse_args()

    train_df, test_df = build_processed_feature_tables(force_rebuild=args.force)
    print(f"Train features shape: {train_df.shape}")
    print(f"Test features shape: {test_df.shape}")


if __name__ == "__main__":
    main()
