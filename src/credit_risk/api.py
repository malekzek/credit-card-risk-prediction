from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import MODEL_PATH

MODEL_FILE = Path(os.getenv("CREDIT_RISK_MODEL_PATH", str(MODEL_PATH))).expanduser()

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Serve default-risk probability predictions from the trained baseline model.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of applicant records to score.",
    )


class PredictResponse(BaseModel):
    model_path: str
    records_scored: int
    expected_feature_count: int
    probabilities: list[float]


def _extract_expected_columns(pipeline: Any) -> list[str]:
    if not hasattr(pipeline, "named_steps"):
        raise ValueError("Loaded model is not a pipeline with named_steps.")

    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None or not hasattr(preprocessor, "transformers"):
        raise ValueError("Pipeline does not contain a ColumnTransformer preprocessor.")

    expected_columns: list[str] = []
    for _, _, columns in preprocessor.transformers:
        if columns is None or columns == "drop":
            continue

        if isinstance(columns, slice):
            raise ValueError("Slice-based feature selection is not supported for API schema export.")

        if isinstance(columns, (pd.Index, np.ndarray)):
            columns = columns.tolist()

        if isinstance(columns, (list, tuple, set)):
            expected_columns.extend([str(col) for col in columns])
        else:
            expected_columns.append(str(columns))

    deduped_columns: list[str] = []
    seen: set[str] = set()
    for col in expected_columns:
        if col not in seen:
            seen.add(col)
            deduped_columns.append(col)

    if not deduped_columns:
        raise ValueError("No expected feature columns were found in the trained model pipeline.")

    return deduped_columns


@lru_cache(maxsize=1)
def _load_model_bundle() -> tuple[Any, list[str]]:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_FILE}. Run train_baseline.py before deploying the API."
        )

    pipeline = joblib.load(MODEL_FILE)
    expected_columns = _extract_expected_columns(pipeline)
    return pipeline, expected_columns


def _build_scoring_frame(records: list[dict[str, Any]], expected_columns: list[str]) -> pd.DataFrame:
    raw_df = pd.DataFrame(records)
    # Reindex to training schema: unknown columns are dropped, missing columns become NaN.
    return raw_df.reindex(columns=expected_columns)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if MODEL_FILE.exists() else "degraded",
        "model_ready": MODEL_FILE.exists(),
        "model_path": str(MODEL_FILE),
    }


@app.get("/schema")
def schema() -> dict[str, Any]:
    try:
        _, expected_columns = _load_model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model schema: {exc}") from exc

    return {
        "expected_feature_count": len(expected_columns),
        "expected_features": expected_columns,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        pipeline, expected_columns = _load_model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc

    scoring_df = _build_scoring_frame(payload.records, expected_columns)

    try:
        proba = pipeline.predict_proba(scoring_df)[:, 1]
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Prediction failed. Ensure provided feature values are compatible with training schema. "
                f"Underlying error: {exc}"
            ),
        ) from exc

    return PredictResponse(
        model_path=str(MODEL_FILE),
        records_scored=int(len(scoring_df)),
        expected_feature_count=int(len(expected_columns)),
        probabilities=[float(x) for x in proba],
    )
