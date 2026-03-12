import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

st.set_page_config(page_title="Temperature Dashboard", layout="wide")
st.title("Temperature Over Time")

openmeteo = openmeteo_requests.Client(
    session=retry(requests_cache.CachedSession(".cache", expire_after=3600), retries=5, backoff_factor=0.2)
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
    r = openmeteo.weather_api(url, params={**params, "latitude": lat, "longitude": lon})[0]
    h = r.Hourly()
    values = h.Variables(0).ValuesAsNumpy()
    dates = pd.date_range(
        pd.to_datetime(h.Time(), unit="s"),
        pd.to_datetime(h.TimeEnd(), unit="s"),
        freq=pd.Timedelta(seconds=h.Interval()),
        inclusive="left",
    )
    return pd.DataFrame({"dt": dates, name: values}).set_index("dt")

# --- Temperature Data ---
df_p = fetch_df(43.67, -70.28, "Portland", temp_params)
df_l = fetch_df(44.100349, -70.214775, "Lewiston", temp_params)
df_s = fetch_df(43.630131, -70.292107, "South Portland", temp_params)

df = df_p.join(df_l, how="inner").join(df_s, how="inner")

# --- Sanity: only allow highlight days that have a full day's worth of data ---
interval_seconds = int((df.index[1] - df.index[0]).total_seconds())
expected_per_day = int(24 * 3600 / interval_seconds)

counts_per_day = df.resample("D").size()
full_days = counts_per_day[counts_per_day == expected_per_day].index

# --- Daily aggregate across ALL towns + ALL hours (sum of hourly temps) ---
daily_aggregate = df.resample("D").sum().sum(axis=1)
daily_aggregate_full = daily_aggregate.loc[full_days]

# Pick 4 coldest and 4 warmest full days
coldest_days = daily_aggregate_full.nsmallest(4).index.normalize()
warmest_days = daily_aggregate_full.nlargest(4).index.normalize()

# --- Temperature Plot ---
fig1 = plt.figure(figsize=(16, 8))
plt.plot(df.index, df["Portland"], label="Portland")
plt.plot(df.index, df["Lewiston"], label="Lewiston")
plt.plot(df.index, df["South Portland"], label="South Portland")

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.grid(True, which="major", axis="x")

for day in coldest_days:
    start = pd.Timestamp(day, tz=df.index.tz)
    end = start + pd.Timedelta(days=1)
    ax.axvspan(start, end, alpha=0.15, color="blue")

for day in warmest_days:
    start = pd.Timestamp(day, tz=df.index.tz)
    end = start + pd.Timedelta(days=1)
    ax.axvspan(start, end, alpha=0.15, color="red")

plt.xlabel("Date")
plt.ylabel("Temperature (°F)")
plt.title("Temperature Over Time (4 Coldest and 4 Warmest Full Days Highlighted)")
plt.legend()
plt.tight_layout()

st.pyplot(fig1)

# --- Cloud Cover Data ---
st.title("Daily Regional Cloud Cover")

cloud_p = fetch_df(43.67, -70.28, "Portland", cloud_params)
cloud_l = fetch_df(44.100349, -70.214775, "Lewiston", cloud_params)
cloud_s = fetch_df(43.630131, -70.292107, "South Portland", cloud_params)

cloud_df = cloud_p.join(cloud_l, how="inner").join(cloud_s, how="inner")

# Daily mean cloud cover per town
daily_cloud = cloud_df.resample("D").mean()

# Regional mean cloud cover across all three towns
regional_cloud = daily_cloud.mean(axis=1)

# --- Cloud Cover Plot ---
fig2 = plt.figure(figsize=(16, 6))
plt.bar(regional_cloud.index, regional_cloud.values)

ax2 = plt.gca()
ax2.xaxis.set_major_locator(mdates.DayLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax2.grid(True, which="major", axis="y")

plt.xlabel("Date")
plt.ylabel("Regional Average Cloud Cover (%)")
plt.title("Daily Regional Cloud Cover (CMP Region)")
plt.tight_layout()

st.pyplot(fig2)
