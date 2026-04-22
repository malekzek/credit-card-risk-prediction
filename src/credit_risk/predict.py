import joblib
import pandas as pd

from .config import MODEL_PATH, PREDICTION_DIR, SAMPLE_SUBMISSION_FILE, SUBMISSION_PATH
from .data import load_test_features


def run_prediction() -> None:
    """Create Kaggle-style submission from the saved baseline model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run train_baseline.py first."
        )

    model = joblib.load(MODEL_PATH)
    df_test = load_test_features(use_secondary_features=True)
    print(f"Prediction table shape: {df_test.shape}")

    proba = model.predict_proba(df_test)[:, 1]

    if SAMPLE_SUBMISSION_FILE.exists():
        submission = pd.read_csv(SAMPLE_SUBMISSION_FILE)
        submission["TARGET"] = proba
    else:
        submission = pd.DataFrame(
            {
                "SK_ID_CURR": df_test["SK_ID_CURR"],
                "TARGET": proba,
            }
        )

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"Saved predictions to: {SUBMISSION_PATH}")


if __name__ == "__main__":
    run_prediction()
