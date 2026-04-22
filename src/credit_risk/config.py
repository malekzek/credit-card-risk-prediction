from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRAIN_FILE = RAW_DATA_DIR / "application_train.csv"
TEST_FILE = RAW_DATA_DIR / "application_test.csv"
SAMPLE_SUBMISSION_FILE = RAW_DATA_DIR / "sample_submission.csv"

BUREAU_FILE = RAW_DATA_DIR / "bureau.csv"
BUREAU_BALANCE_FILE = RAW_DATA_DIR / "bureau_balance.csv"
CREDIT_CARD_BALANCE_FILE = RAW_DATA_DIR / "credit_card_balance.csv"
INSTALLMENTS_FILE = RAW_DATA_DIR / "installments_payments.csv"
POS_CASH_FILE = RAW_DATA_DIR / "POS_CASH_balance.csv"
PREVIOUS_APPLICATION_FILE = RAW_DATA_DIR / "previous_application.csv"
COLUMN_DESCRIPTION_FILE = RAW_DATA_DIR / "HomeCredit_columns_description.csv"

MODEL_DIR = PROJECT_DIR / "outputs" / "models"
PREDICTION_DIR = PROJECT_DIR / "outputs" / "predictions"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_PATH = MODEL_DIR / "baseline_hgb.joblib"
METRICS_PATH = REPORT_DIR / "baseline_metrics.json"
SUBMISSION_PATH = PREDICTION_DIR / "submission_baseline.csv"
MODEL_COMPARISON_JSON_PATH = REPORT_DIR / "model_comparison.json"
MODEL_COMPARISON_CSV_PATH = REPORT_DIR / "model_comparison.csv"
EXPLAINABILITY_JSON_PATH = REPORT_DIR / "explainability_summary.json"
EXPLAINABILITY_TOP_FEATURES_CSV_PATH = REPORT_DIR / "explainability_top_features.csv"
EXPLAINABILITY_PLOT_PATH = REPORT_DIR / "figures" / "shap_summary_bar.png"

SECONDARY_FEATURES_FILE = PROCESSED_DATA_DIR / "secondary_features.pkl"
TRAIN_FEATURES_FILE = PROCESSED_DATA_DIR / "application_train_features.pkl"
TEST_FEATURES_FILE = PROCESSED_DATA_DIR / "application_test_features.pkl"

RANDOM_STATE = 42
