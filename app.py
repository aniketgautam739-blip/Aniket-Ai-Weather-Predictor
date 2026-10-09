import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="SmartGrid AI", page_icon="⚡", layout="wide")

st.title("⚡ SmartGrid AI")
st.subheader("Interactive next-hour electricity demand forecasting")
st.write("Train a machine-learning model on hourly consumption data, inspect its performance, and change the latest demand inputs to explore a new forecast.")

with st.sidebar:
    st.header("1. Choose your data")
    uploaded = st.file_uploader("Upload hourly CSV (optional)", type=["csv"])
    st.caption("CSV needs a timestamp column and a numeric consumption column. If you don't upload a file, the app uses clearly labelled simulated demo data.")
    use_demo = st.checkbox("Use simulated demo data if no CSV is uploaded", value=True)

def make_demo_data():
    rng = np.random.default_rng(12)
    n = 24 * 120
    ts = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=n, freq="h")
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    daily = 22*np.sin((hour-7)*2*np.pi/24) + 14*np.sin((hour-17)*2*np.pi/24)
    work = np.where(dow < 5, 15, -4)
    trend = np.linspace(0, 8, n)
    noise = rng.normal(0, 5, n)
    consumption = 180 + daily + work + trend + noise
    consumption = np.maximum(consumption, 40)
    return pd.DataFrame({"timestamp": ts, "consumption": consumption})

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Couldn't read this CSV: {e}")
        st.stop()
    st.sidebar.subheader("2. Match the columns")
    cols = list(raw.columns)
    default_time = next((i for i,c in enumerate(cols) if any(x in c.lower() for x in ["time","date","datetime","timestamp"])), 0)
    default_val = next((i for i,c in enumerate(cols) if any(x in c.lower() for x in ["consumption","load","demand","usage","mw","kwh","value"])), min(1, len(cols)-1))
    time_col = st.sidebar.selectbox("Timestamp column", cols, index=default_time)
    value_col = st.sidebar.selectbox("Consumption column", cols, index=default_val)
    df = pd.DataFrame({"timestamp": pd.to_datetime(raw[time_col], errors="coerce"),
                       "consumption": pd.to_numeric(raw[value_col], errors="coerce")})
    df = df.dropna().sort_values("timestamp")
    # Convert to hourly observations, averaging duplicate timestamps within an hour.
    df["timestamp"] = df["timestamp"].dt.floor("h")
    df = df.groupby("timestamp", as_index=False)["consumption"].mean().sort_values("timestamp")
    data_source = f"Uploaded CSV: {uploaded.name}"
elif use_demo:
    df = make_demo_data()
    data_source = "Simulated demo data (not real utility measurements)"
else:
    st.warning("Upload a CSV or enable simulated demo data in the sidebar.")
    st.stop()

if len(df) < 60:
    st.error("Please provide at least 60 hourly observations. A few days of hourly data is the minimum for this demonstration; more data is better.")
    st.stop()

df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["lag_1"] = df["consumption"].shift(1)
df["lag_2"] = df["consumption"].shift(2)
df["lag_3"] = df["consumption"].shift(3)
df["lag_24"] = df["consumption"].shift(24)
df["lag_48"] = df["consumption"].shift(48)
features = ["hour", "dayofweek", "lag_1", "lag_2", "lag_3", "lag_24", "lag_48"]
model_data = df.dropna(subset=features + ["consumption"]).copy()
if len(model_data) < 30:
    st.error("Not enough usable rows after creating lag features. Please upload more hourly data.")
    st.stop()

split = max(int(len(model_data) * 0.8), 1)
train = model_data.iloc[:split]
test = model_data.iloc[split:]
if len(test) < 2:
    st.error("Please upload more rows so the app can evaluate on a held-out test period.")
    st.stop()

model = RandomForestRegressor(n_estimators=160, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=-1)
model.fit(train[features], train["consumption"])
pred_test = model.predict(test[features])
baseline = test["lag_1"].to_numpy()
mae = mean_absolute_error(test["consumption"], pred_test)
rmse = float(np.sqrt(mean_squared_error(test["consumption"], pred_test)))
base_mae = mean_absolute_error(test["consumption"], baseline)

st.caption(f"Data source: **{data_source}** · {len(df):,} hourly records · {df['timestamp'].min():%Y-%m-%d} to {df['timestamp'].max():%Y-%m-%d}")
if "Simulated" in data_source:
    st.info("Demo mode: the data below is simulated for learning. Upload real hourly data before claiming real-world forecasting results.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Hourly records", f"{len(df):,}")
m2.metric("Test MAE", f"{mae:.2f}", help="Average absolute prediction error on the later test period, in the dataset's consumption units.")
m3.metric("Test RMSE", f"{rmse:.2f}", help="Root mean squared error on the later test period.")
m4.metric("Previous-hour baseline MAE", f"{base_mae:.2f}", help="Simple benchmark: predict that the next hour equals the previous hour.")

st.header("2. Explore historical consumption")
history_hours = st.slider("How much history to display", min_value=24, max_value=min(24*30, len(df)), value=min(24*7, len(df)), step=24)
hist = df.tail(history_hours)
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["timestamp"], y=hist["consumption"], mode="lines", name="Historical consumption"))
fig.update_layout(xaxis_title="Time", yaxis_title="Consumption (dataset units)", height=370, margin=dict(l=10,r=10,t=25,b=10), legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

st.header("3. Interactively predict the next hour")
st.write("The default values come from the most recent observations. Change them to explore a what-if forecast, then click the button.")
latest = df.iloc[-1]
def get_lag(k):
    return float(df.iloc[-k]["consumption"]) if len(df) >= k else float(df["consumption"].iloc[-1])

with st.form("forecast_form"):
    a,b,c = st.columns(3)
    lag1 = a.number_input("Latest hour (lag 1)", value=float(get_lag(1)), min_value=0.0, step=1.0, help="Most recent observed consumption.")
    lag2 = b.number_input("2 hours ago (lag 2)", value=float(get_lag(2)), min_value=0.0, step=1.0)
    lag3 = c.number_input("3 hours ago (lag 3)", value=float(get_lag(3)), min_value=0.0, step=1.0)
    d,e,f = st.columns(3)
    lag24 = d.number_input("Same hour yesterday (lag 24)", value=float(get_lag(24)), min_value=0.0, step=1.0)
    lag48 = e.number_input("Same hour 2 days ago (lag 48)", value=float(get_lag(48)), min_value=0.0, step=1.0)
    next_time = latest["timestamp"] + pd.Timedelta(hours=1)
    forecast_hour = f.selectbox("Forecast hour", options=list(range(24)), index=int(next_time.hour), format_func=lambda h: f"{h:02d}:00")
    forecast_day = st.selectbox("Day of week", options=list(range(7)), index=int(next_time.dayofweek), format_func=lambda d: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][d])
    submitted = st.form_submit_button("⚡ Predict next-hour consumption", type="primary", use_container_width=True)

if submitted:
    Xnew = pd.DataFrame([{"hour": forecast_hour, "dayofweek": forecast_day, "lag_1": lag1, "lag_2": lag2,
                          "lag_3": lag3, "lag_24": lag24, "lag_48": lag48}])[features]
    prediction = float(model.predict(Xnew)[0])
    st.success("Forecast generated by the trained Random Forest model.")
    p1,p2 = st.columns(2)
    p1.metric("Predicted consumption", f"{prediction:,.2f}", help="Units are the same as your uploaded data.")
    p2.metric("Previous-hour baseline", f"{lag1:,.2f}")
    plot_df = pd.DataFrame({"Time": list(hist["timestamp"].tail(24)) + [next_time],
                            "Consumption": list(hist["consumption"].tail(24)) + [prediction],
                            "Type": ["Historical"] * min(24, len(hist)) + ["AI forecast"]})
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=hist["timestamp"].tail(24), y=hist["consumption"].tail(24), mode="lines+markers", name="Recent actual/demo data"))
    chart.add_trace(go.Scatter(x=[next_time], y=[prediction], mode="markers", marker=dict(size=13, symbol="star"), name="Next-hour AI forecast"))
    chart.update_layout(xaxis_title="Time", yaxis_title="Consumption (dataset units)", height=350, margin=dict(l=10,r=10,t=25,b=10))
    st.plotly_chart(chart, use_container_width=True)
    st.caption("Changing inputs creates a what-if forecast. It does not change the historical dataset or retrain the model.")

st.header("4. Understand the model")
importance = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance")
imp_fig = go.Figure(go.Bar(x=importance["Importance"], y=importance["Feature"], orientation="h"))
imp_fig.update_layout(xaxis_title="Relative feature importance", yaxis_title="", height=300, margin=dict(l=10,r=10,t=15,b=10))
st.plotly_chart(imp_fig, use_container_width=True)
st.download_button("Download cleaned hourly data (CSV)", data=df[["timestamp","consumption"]].to_csv(index=False).encode("utf-8"),
                   file_name="hourly_consumption_cleaned.csv", mime="text/csv")

with st.expander("Methodology and limitations"):
    st.markdown("""
    - Model: Random Forest regression using hour, day of week, and recent consumption lags (1, 2, 3, 24, and 48 hours).
    - Evaluation: chronological 80/20 split; the model is evaluated on the later portion of the data.
    - Baseline: predict the next hour using the previous hour's consumption.
    - Forecast horizon: one hour ahead.
    - Data quality: the model assumes the rows represent hourly observations. Missing hours are not automatically filled.
    - This is an educational prototype, not a utility-grade forecasting or grid-control system. A simulated dataset is not evidence of real-world accuracy.
    """)
