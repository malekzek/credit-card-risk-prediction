# Credit Risk Prediction (Beginner Starter)

This is a beginner-friendly machine learning starter project built on the Home Credit CSV files.

## What this version does

- Trains a **baseline credit risk model** using `data/raw/application_train.csv`
- Builds applicant-level aggregated features from secondary tables (`bureau`, `previous_application`, `installments`, `POS_CASH`, `credit_card`, `bureau_balance`)
- Evaluates the model using **ROC-AUC**
- Saves trained model to `outputs/models/`
- Generates prediction file for `data/raw/application_test.csv`

> Note: Training and prediction now use cached aggregated features from secondary tables.

## Project structure

```text
home-credit-default-risk/
  data/
    raw/
      application_train.csv
      application_test.csv
      ... other source CSV files ...
    processed/
      (future engineered datasets)
  notebooks/
    01_starter_overview.ipynb
  src/
    credit_risk/
      config.py
      data.py
      features.py
      secondary_features.py
      train.py
      predict.py
  outputs/
    models/
    predictions/
  reports/
  build_aggregated_features.py
  train_baseline.py
  predict_baseline.py
  requirements.txt
```

## Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run API locally

Train the baseline model first (required once):

```bash
python train_baseline.py
```

Then start the API:

```bash
uvicorn serve_api:app --host 0.0.0.0 --port 8000 --reload
```

Useful endpoints:
- `GET /health`
- `GET /schema`
- `POST /predict`

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"records":[{"EXT_SOURCE_2":0.7,"EXT_SOURCE_3":0.6,"AMT_CREDIT":500000}]}'
```

## Deploy API with Docker

Build image:

```bash
docker build -t credit-risk-api .
```

Run container:

```bash
docker run --rm -p 8000:8000 credit-risk-api
```

Optional model override path:

```bash
docker run --rm -p 8000:8000 \
  -e CREDIT_RISK_MODEL_PATH=/app/outputs/models/baseline_hgb.joblib \
  credit-risk-api
```

## Run training

Optional (build feature caches explicitly):

```bash
python build_aggregated_features.py --force
```

Then train:

```bash
python train_baseline.py
```

This will:
- print fold-wise and mean ROC-AUC from stratified cross-validation
- save model to `outputs/models/baseline_hgb.joblib`
- save metrics to `reports/baseline_metrics.json`

## Run prediction

```bash
python predict_baseline.py
```

This will save:
- `outputs/predictions/submission_baseline.csv`

## Compare models (HGB vs LightGBM vs CatBoost)

```bash
python compare_models.py
```

This will save:
- `reports/model_comparison.json`
- `reports/model_comparison.csv`

## Generate SHAP explainability report

```bash
python generate_explainability_report.py
```

By default, this runs the CatBoost explainer.

Optional (choose explainability backend):

```bash
python generate_explainability_report.py --model catboost
python generate_explainability_report.py --model lightgbm
python generate_explainability_report.py --model auto
```

Notes:
- `--model auto` picks the winner from `reports/model_comparison.csv`.
- If auto selects an unsupported explainer (for example `hist_gradient_boosting`), it falls back to CatBoost.

This will save:
- `reports/explainability_summary.json`
- `reports/explainability_top_features.csv`
- `reports/figures/shap_summary_bar.png`

## Run data quality report (next step)

```bash
python generate_data_report.py
```

This will save:
- `reports/data_quality_report.json`
- `reports/application_train_missingness.csv`
- `reports/application_train_dtypes.csv`

## Should this be a notebook project?

For data science, using notebooks is a good idea for exploration and learning.

- Use notebooks for EDA, quick experiments, and visualizations.
- Use Python modules/scripts in `src/` for reusable training and prediction pipelines.

This project uses both: notebook for learning + scripts for reliable runs.

## Beginner notes

- `TARGET = 1` means higher repayment/default risk in this dataset context.
- ROC-AUC closer to `1.0` is better; `0.5` means random-like ranking.
- This baseline is intentionally simple so you can understand each step.

## Next planned improvements

1. Add hyperparameter tuning and experiment tracking.
2. Add model monitoring and drift checks.

## Push to GitHub

```bash
git init
git add .
git commit -m "Add credit risk pipeline, explainability, and deployment API"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```
