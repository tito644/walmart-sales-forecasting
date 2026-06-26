# 🛒 Walmart Sales Forecasting & Optimization

> End-to-end time-series forecasting project: from raw retail data to a deployed, interactive sales prediction app.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

🔗 **Live Demo:** _[Streamlit app link — add after deployment]_
📊 **Kaggle Notebook:** _[Kaggle notebook link — add after publishing]_

---

## 📌 Project Overview

This project predicts weekly sales for 45 Walmart stores using historical sales data, economic indicators (CPI, Unemployment, Fuel Price), and holiday effects. The goal is to provide a forecasting tool that helps optimize inventory, marketing, and staffing decisions.

**Dataset:** [Walmart Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/walmart-dataset)

---

## 🎯 Key Results

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| Naive Baseline | — | — | — |
| Prophet | — | — | — |
| SARIMA | — | — | — |
| XGBoost | — | — | — |

> _Table will be filled in after Milestone 3. Best model highlighted in **bold**._

---

## 🗂️ Project Structure

```
walmart-sales-forecasting/
├── data/
│   ├── raw/              # Original, untouched dataset
│   └── processed/        # Cleaned & feature-engineered data
├── notebooks/            # EDA, modeling, experimentation notebooks
├── src/                  # Reusable Python modules (cleaning, features, models)
├── app/                  # Streamlit application
├── models/               # Saved/serialized final model(s)
├── reports/
│   └── figures/          # Exported charts for the report/README
├── mlruns/                # MLflow experiment tracking (gitignored if large)
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/walmart-sales-forecasting.git
cd walmart-sales-forecasting

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app/app.py
```

---

## 🧭 Methodology (Milestones)

1. **Data Collection, Exploration & Preprocessing** — EDA, missing values, time-based feature engineering
2. **Data Analysis & Visualization** — correlation analysis, seasonality decomposition, interactive dashboard
3. **Forecasting Model Development & Optimization** — Prophet, SARIMA, XGBoost compared via time-series CV
4. **MLOps, Deployment & Monitoring** — MLflow tracking, Streamlit deployment, drift monitoring plan
5. **Final Documentation & Presentation** — full report + stakeholder presentation

Detailed write-up for each milestone lives in `notebooks/` and `reports/`.

---

## 🛠️ Tech Stack

`Python` `Pandas` `Scikit-learn` `Prophet` `XGBoost` `Statsmodels` `Plotly` `Streamlit` `MLflow`

---

## 📈 Business Value

Accurate weekly sales forecasts allow retail teams to:
- Optimize inventory levels and reduce stockouts/overstock
- Plan staffing and promotions around predicted demand peaks (holidays)
- Detect early warning signs of demand shifts via monitoring

---

## 👤 Author

**Tareq** — Data Scientist
[LinkedIn](#) · [Kaggle](#) · [GitHub](#)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
