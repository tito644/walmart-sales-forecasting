"""
app.py
Streamlit app for interactive Walmart weekly sales forecasting.
Run with: streamlit run app/app.py
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Walmart Sales Forecast", layout="wide")

st.title("🛒 Walmart Weekly Sales Forecast")
st.markdown(
    "Predict weekly sales for a given store based on economic indicators "
    "and holiday effects. Built as part of an end-to-end forecasting project."
)

# --- Sidebar inputs ---
st.sidebar.header("Input Parameters")
store_id = st.sidebar.selectbox("Store ID", options=list(range(1, 46)))
is_holiday = st.sidebar.selectbox("Holiday Week?", options=["No", "Yes"])
temperature = st.sidebar.slider("Temperature (°F)", min_value=-10, max_value=110, value=60)
fuel_price = st.sidebar.slider("Fuel Price ($)", min_value=2.0, max_value=5.0, value=3.5, step=0.1)
cpi = st.sidebar.number_input("CPI", value=210.0)
unemployment = st.sidebar.number_input("Unemployment Rate (%)", value=7.5)

# --- Prediction placeholder ---
# TODO: load the trained model from /models and replace this mock logic
if st.sidebar.button("Predict Sales"):
    # mock_prediction = model.predict(...)
    mock_prediction = 1_000_000  # placeholder
    st.metric(label="Predicted Weekly Sales", value=f"${mock_prediction:,.0f}")
    st.info("⚠️ This is a placeholder. Connect the trained model from `models/` to get real predictions.")

st.markdown("---")
st.caption("Project by Tareq · Data Science Portfolio")
