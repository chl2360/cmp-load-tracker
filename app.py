import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import openmeteo_requests
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="CMP Load Tracker", layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>Proprietary CMP Load Tracker</h1>",
    unsafe_allow_html=True
)

# -------------------------
# LOAD + TRAIN ML MODEL
# -------------------------
@st.cache_data
def load_training_data(file_name="for_ml.csv"):
    df = pd.read_csv(file_name)
    df = df[["Hour", "Temperature", "load"]].dropna().reset_index(drop=True)
    return df

@st.cache_resource
def train_model(file_name="for_ml.csv"):
    train_df = load_training_data(file_name).copy()

    # Treat Hour as cyclical
    train_df["Hour_sin"] = np.sin(2 * np.pi * train_df["Hour"] / 24)
    train_df["Hour_cos"] = np.cos(2 * np.pi * train_df["Hour"] / 24)

    X = train_df[["Hour_sin", "Hour_cos", "Temperature"]]
    y = train_df["load"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    model.fit(X, y)

    # In-sample diagnostics
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    metrics = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }

    return model, metrics

def prepare_features(input_df):
    temp_df = input_df.copy()
    temp_df["Hour_sin"] = np.sin(2 * np.pi * temp_df["Hour"] / 24)
    temp_df["Hour_cos"] = np.cos(2 * np.pi * temp_df["Hour"] / 24)
    return temp_df[["Hour_sin", "Hour_cos", "Temperature"]]

model, metrics = train_model("for_ml.csv")

# -------------------------
# OPEN-METEO CLIENT
# -------------------------
openmeteo = openmeteo_requests.Client(
    session=retry(
        requests_cache.CachedSession(".cache", expire_after=3600),
        retries=5,
        backoff_factor=0.2
    )
)

url = "https://api.open-meteo.com/v1/forecast"

temp_params = {
    "hourly": "temperature_2m",
    "timezone": "America/New_York",
    "past_days": 5,
    "forecast_days": 14,
    "temperature_unit": "fahrenheit",
}

cloud_params = {
    "hourly": "cloud_cover",
    "timezone": "America/New_York",
    "forecast_days": 14,
}

def fetch_df(lat, lon, name, params):
    response = openmeteo.weather_api(
        url,
        params={**params, "latitude": lat, "longitude": lon}
    )[0]

    hourly = response.Hourly()
    values = hourly.Variables(0).ValuesAsNumpy()

    dates = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ).tz_convert("America/New_York")

    return pd.DataFrame({"dt": dates, name: values}).set_index("dt")

# -------------------------
# TEMPERATURE DATA
# -------------------------
df_p = fetch_df(43.67, -70.28, "Portland", temp_params)
df_l = fetch_df(44.100349, -70.214775, "Lewiston", temp_params)
df_s = fetch_df(43.630131, -70.292107, "South Portland", temp_params)

df = df_p.join(df_l, how="inner").join(df_s, how="inner")

# -------------------------
# FULL DAY SANITY CHECK
# -------------------------
interval_seconds = int((df.index[1] - df.index[0]).total_seconds())
expected_per_day = int(24 * 3600 / interval_seconds)

counts_per_day = df.resample("D").size()
full_days = counts_per_day[counts_per_day == expected_per_day].index

# -------------------------
# DAILY AGGREGATE / EXTREME DAYS
# -------------------------
daily_aggregate = df.resample("D").sum().sum(axis=1)
daily_aggregate_full = daily_aggregate.loc[full_days]

coldest_days = daily_aggregate_full.nsmallest(4).index.normalize()
warmest_days = daily_aggregate_full.nlargest(4).index.normalize()

# -------------------------
# TEMPERATURE PLOT
# -------------------------
fig1 = plt.figure(figsize=(16, 8))
plt.plot(df.index, df["Portland"], label="Portland")
plt.plot(df.index, df["Lewiston"], label="Lewiston")
plt.plot(df.index, df["South Portland"], label="South Portland")

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.grid(True, which="major", axis="x")

for day in coldest_days:
    start = day
    end = start + pd.Timedelta(days=1)
    ax.axvspan(start, end, alpha=0.15, color="blue")

for day in warmest_days:
    start = day
    end = start + pd.Timedelta(days=1)
    ax.axvspan(start, end, alpha=0.15, color="red")

plt.xlabel("Date")
plt.ylabel("Temperature (°F)")
plt.title("Temperature Forecast")
plt.legend()
plt.tight_layout()

st.pyplot(fig1)

# -------------------------
# CLOUD COVER DATA
# -------------------------
cloud_p = fetch_df(43.67, -70.28, "Portland", cloud_params)
cloud_l = fetch_df(44.100349, -70.214775, "Lewiston", cloud_params)
cloud_s = fetch_df(43.630131, -70.292107, "South Portland", cloud_params)

cloud_df = cloud_p.join(cloud_l, how="inner").join(cloud_s, how="inner")

daily_cloud = cloud_df.resample("D").mean()
regional_cloud = daily_cloud.mean(axis=1)

# -------------------------
# CLOUD COVER PLOT
# -------------------------
fig2 = plt.figure(figsize=(16, 6))
plt.bar(regional_cloud.index, regional_cloud.values)

ax2 = plt.gca()
ax2.xaxis.set_major_locator(mdates.DayLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax2.grid(True, which="major", axis="y")

plt.xlabel("Date")
plt.ylabel("Regional Average Cloud Cover (%)")
plt.title("Regional Cloud Cover Forecast")
plt.tight_layout()

st.pyplot(fig2)

# -------------------------
# BUILD ML INPUTS FROM FORECASTED TEMPERATURES
# -------------------------
forecast_df = df.copy()

# Regional hourly average temperature from the 3 locations
forecast_df["Temperature"] = forecast_df[["Portland", "Lewiston", "South Portland"]].mean(axis=1)

# Convert forecast timestamps to Hour Ending 1-24
forecast_df["Hour"] = forecast_df.index.hour + 1

ml_input = forecast_df[["Hour", "Temperature"]].copy()

# -------------------------
# PREDICT LOAD
# -------------------------
X_future = prepare_features(ml_input)
forecast_df["Predicted_Load"] = model.predict(X_future)

# -------------------------
# HOURLY PREDICTED LOAD PLOT
# -------------------------
fig3 = plt.figure(figsize=(16, 6))
plt.plot(forecast_df.index, forecast_df["Predicted_Load"], linewidth=2)

ax3 = plt.gca()
ax3.xaxis.set_major_locator(mdates.DayLocator())
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax3.grid(True, which="major", axis="both")

plt.xlabel("Date")
plt.ylabel("Predicted Load")
plt.title("Predicted Hourly Load from Forecasted Temperature")
plt.tight_layout()

st.pyplot(fig3)

# -------------------------
# DAILY AGGREGATE PREDICTED LOAD
# -------------------------
daily_predicted_load = forecast_df["Predicted_Load"].resample("D").sum()

fig4 = plt.figure(figsize=(16, 6))
plt.bar(daily_predicted_load.index, daily_predicted_load.values)

ax4 = plt.gca()
ax4.xaxis.set_major_locator(mdates.DayLocator())
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax4.grid(True, which="major", axis="y")

plt.xlabel("Date")
plt.ylabel("Daily Predicted Load")
plt.title("Daily Aggregate Predicted Load")
plt.tight_layout()

st.pyplot(fig4)

# -------------------------
# MODEL DIAGNOSTICS
# -------------------------
st.subheader("Model Diagnostics")
st.write(metrics)

# -------------------------
# DISPLAY TABLE
# -------------------------
st.subheader("Hourly Predicted Load")

display_df = forecast_df[[
    "Portland",
    "Lewiston",
    "South Portland",
    "Temperature",
    "Hour",
    "Predicted_Load"
]].copy()

display_df = display_df.rename(columns={
    "Temperature": "Regional_Avg_Temperature"
})

st.dataframe(display_df)

# -------------------------
# CSV DOWNLOAD
# -------------------------
csv_export_df = display_df.reset_index().rename(columns={"dt": "Datetime_ET"})
csv_data = csv_export_df.to_csv(index=False)

st.download_button(
    label="Download Predicted Load CSV",
    data=csv_data,
    file_name="predicted_hourly_load.csv",
    mime="text/csv"
)
