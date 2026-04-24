# Credit Risk Prediction Portfolio Project (Home Credit)

End-to-end machine learning system for credit default risk prediction using the Home Credit dataset. This project combines multi-table feature engineering, model evaluation, explainability, and containerized API deployment.

## Project Snapshot

- Problem: binary classification for default risk ranking
- Label meaning: TARGET = 1 indicates higher risk (payment difficulty), TARGET = 0 indicates lower risk
- Data scale (all raw CSVs combined): 58,538,856 rows across 10 files and 346 columns summed across tables
- Tech stack: Python, pandas, scikit-learn, LightGBM, CatBoost, SHAP-style feature attribution, FastAPI, Docker

## Business Objective

Build a reproducible scoring pipeline that can estimate customer default risk and expose predictions through an API suitable for integration into downstream systems.

## What Was Built

1. Data quality profiling across raw source tables.
2. Applicant-level feature engineering with aggregated secondary-table features.
3. Baseline training and cross-validation workflow.
4. Model comparison across HistGradientBoosting, LightGBM, and CatBoost.
5. Explainability artifacts and ranked feature impact report.
6. FastAPI inference service and Dockerized deployment.
7. Portfolio-ready matplotlib figures for project communication.

## Results

- Baseline model with secondary aggregates:
  - Validation ROC-AUC: 0.7642
  - 3-fold CV ROC-AUC: 0.7642 +/- 0.0007
  - Train rows used: 120,000
- Model comparison run (40,000-row CV subset):
  - CatBoost: 0.7570
  - LightGBM: 0.7503
  - HistGradientBoosting: 0.7491

## Project Figures

Pipeline overview:

![Pipeline overview](reports/figures/pipeline_overview.png)

Model comparison (3-fold CV ROC-AUC):

![Model comparison ROC-AUC](reports/figures/model_comparison_auc.png)

Data quality overview (class balance + missingness):

![Data quality overview](reports/figures/data_quality_overview.png)

Explainability summary (mean absolute feature impact):

![SHAP summary bar](reports/figures/shap_summary_bar.png)

## Repository Layout

```text
home-credit-default-risk/
  data/
    raw/                  # Original Home Credit CSV files (not committed)
    processed/            # Cached engineered tables (not committed)
  notebooks/
    01_starter_overview.ipynb
  outputs/
    models/               # Trained model artifacts (not committed)
    predictions/          # Prediction/submission files (not committed)
  reports/                # Generated reports and selected committed figures
    figures/
  src/
    credit_risk/
      api.py
      config.py
      data.py
      data_quality.py
      explainability.py
      features.py
      model_compare.py
      predict.py
      report_figures.py
      secondary_features.py
      train.py
  build_aggregated_features.py
  compare_models.py
  generate_data_report.py
  generate_explainability_report.py
  generate_project_figures.py
  predict_baseline.py
  serve_api.py
  train_baseline.py
  Dockerfile
  requirements.txt
```

## Reproduce Artifacts

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full workflow:

```bash
python build_aggregated_features.py --force
python train_baseline.py
python predict_baseline.py
python compare_models.py
python generate_explainability_report.py --model catboost
python generate_data_report.py
python generate_project_figures.py
```

Primary outputs:

- outputs/models/baseline_hgb.joblib
- outputs/predictions/submission_baseline.csv
- reports/baseline_metrics.json
- reports/model_comparison.json
- reports/explainability_summary.json
- reports/explainability_top_features.csv
- reports/figures/pipeline_overview.png
- reports/figures/model_comparison_auc.png
- reports/figures/data_quality_overview.png
- reports/figures/shap_summary_bar.png

## API Inference

Start the API locally:

```bash
uvicorn serve_api:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:

- GET /health
- GET /schema
- POST /predict

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"records":[{"EXT_SOURCE_2":0.7,"EXT_SOURCE_3":0.6,"AMT_CREDIT":500000}]}'
```

## Docker Deployment

Build image:

```bash
docker build -t credit-risk-api .
```

Run container:

```bash
docker run --rm -p 8000:8000 credit-risk-api
```

Optional model override:

```bash
docker run --rm -p 8000:8000 \
  -e CREDIT_RISK_MODEL_PATH=/app/outputs/models/baseline_hgb.joblib \
  credit-risk-api
```

## Notes

- Raw data files are intentionally excluded from Git due to size and licensing constraints.
- Generated model binaries and prediction outputs are excluded from version control.
- Selected figure images are committed for portfolio presentation.
