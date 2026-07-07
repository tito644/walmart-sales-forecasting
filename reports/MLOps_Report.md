# MLOps Report

This document explains how experiment tracking, model versioning, deployment, and monitoring were handled for this project (Milestone 4).

## 1. Experiment Tracking — MLflow

All model runs (baselines, Prophet, XGBoost) are logged with MLflow inside `notebooks/03_modeling.ipynb`:

- **Tracking backend:** SQLite (`mlflow.db` in the project root)
- **Experiment name:** `walmart-sales-forecasting`
- **What's logged per run:**
  - Parameters (e.g. XGBoost's tuned hyperparameters: `n_estimators`, `max_depth`, `learning_rate`)
  - Metrics: RMSE, MAE, MAPE
  - The final XGBoost model artifact (logged with `mlflow.xgboost.log_model`)

To view the runs locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://localhost:5000` in a browser to compare all logged runs side by side.

## 2. Model Versioning

For this project's scope, the final trained model is saved directly with `joblib` into `models/xgb_sales_model.pkl`, along with the exact list of features it expects (`models/model_features.pkl`). This keeps the Streamlit app simple — it just loads these two files at startup.

A natural next step for a larger team setup would be to add **DVC** (Data Version Control) to version the dataset and model files alongside Git, so changes to the data or model can be tracked the same way as code changes. This wasn't strictly necessary at this project's scale (one dataset, one model), but is a planned improvement — see "Future Improvements" below.

## 3. Deployment

The model is deployed as an interactive **Streamlit** app (`app/app.py`):

- Loads the trained model + feature list once (cached with `@st.cache_resource`)
- Lets the user pick a store and adjust "what-if" conditions (month, holiday flag, temperature, fuel price, CPI, unemployment)
- Automatically pulls the store's real recent sales history to build the lag features (`lag_1`, `lag_52`, `rolling_mean_4`) instead of asking the user to guess them
- Returns a predicted weekly sales figure and compares it to the store's historical average

Run locally with:
```bash
streamlit run app/app.py
```

(Optional) Deployed live at: `[add Streamlit Community Cloud link after deployment]`

## 4. Model Monitoring (plan)

This project doesn't have live production traffic, so monitoring is described here as a plan rather than a running system:

- **Performance tracking:** log every prediction with its inputs to a simple table/file, and once actual sales for that week are known, compute the rolling MAPE over the last N predictions.
- **Drift detection:** compare the distribution of incoming feature values (e.g. CPI, Unemployment) against the training distribution. A large shift (e.g. CPI moving far outside the training range of 126–227) would be a signal to retrain.
- **Alerting threshold:** if rolling MAPE exceeds a chosen threshold (e.g. 8%, roughly double the test MAPE of ~4.2%), that would trigger a retraining review.
- **Retraining cadence:** since this is weekly retail data, a reasonable cadence would be to retrain monthly or quarterly as new weeks of actual sales become available, then re-run the same time-based evaluation in `03_modeling.ipynb` before promoting a new model version.

## 5. Future Improvements

- Add DVC for dataset/model versioning
- Automate retraining with a scheduled pipeline (e.g. GitHub Actions on a cron schedule)
- Add the monitoring plan above as an actual lightweight logging script
- Try LSTM/deep learning models for comparison once more historical data is available (the current ~2.7 years is on the shorter side for deep learning approaches)
