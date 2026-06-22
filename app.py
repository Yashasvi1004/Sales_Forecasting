import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Rossmann Sales Forecast Dashboard", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('xgb_model.pkl')

@st.cache_data
def load_data():
    preds = pd.read_csv('val_predictions.csv', parse_dates=['Date'])
    importance = pd.read_csv('feature_importance.csv')
    latest_features = pd.read_csv('latest_store_features.csv')
    future_preds = pd.read_csv('future_predictions.csv', parse_dates=['Date'])
    return preds, importance, latest_features, future_preds

model = load_model()
preds, importance, latest_features, future_preds = load_data()

st.title("📊 Rossmann Store Sales — Forecast Dashboard")
st.markdown("XGBoost-based demand forecasting across 1,115 stores")

store_list = sorted(preds['Store'].unique())
selected_store = st.sidebar.selectbox("Select Store", store_list)

store_data = preds[preds['Store'] == selected_store].sort_values('Date')

mae = np.mean(np.abs(store_data['Actual'] - store_data['Predicted']))
rmse = np.sqrt(np.mean((store_data['Actual'] - store_data['Predicted'])**2))
mape = np.mean(np.abs((store_data['Actual'] - store_data['Predicted']) / store_data['Actual'])) * 100

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:,.0f}")
col2.metric("RMSE", f"{rmse:,.0f}")
col3.metric("MAPE", f"{mape:.2f}%")

st.subheader(f"Actual vs Predicted Sales — Store {selected_store}")
fig = px.line(store_data, x='Date', y=['Actual', 'Predicted'])
st.plotly_chart(fig, use_container_width=True)

st.subheader("Top Features Driving Predictions")
fig2 = px.bar(importance.head(10), x='Importance', y='Feature', orientation='h')
fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig2, use_container_width=True)

st.subheader("🔧 What-If Simulator: Promo Impact")
row = latest_features[latest_features['Store'] == selected_store].copy()

if not row.empty:
    promo_off = row.copy()
    promo_off['Promo'] = 0
    promo_on = row.copy()
    promo_on['Promo'] = 1

    pred_off = model.predict(promo_off)[0]
    pred_on = model.predict(promo_on)[0]

    colA, colB = st.columns(2)
    colA.metric("Predicted Sales (No Promo)", f"{pred_off:,.0f}")
    colB.metric("Predicted Sales (With Promo)", f"{pred_on:,.0f}", delta=f"{pred_on - pred_off:,.0f}")
else:
    st.warning("No feature data available for this store.")

st.subheader(f"🔮 Future Sales Forecast — Store {selected_store}")

future_store = future_preds[future_preds['Store'] == selected_store].sort_values('Date')

fig3 = px.line(future_store, x='Date', y='Sales_Predicted', title=f'Forecasted Sales (Aug–Sep 2015) — Store {selected_store}')
st.plotly_chart(fig3, use_container_width=True)