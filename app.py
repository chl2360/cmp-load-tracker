# -------------------------
# TEMPERATURE + PREDICTED LOAD OVERLAY PLOT
# -------------------------
fig1, ax1 = plt.subplots(figsize=(16, 8))

# Temperature lines on left axis
line1, = ax1.plot(df.index, df["Portland"], label="Portland")
line2, = ax1.plot(df.index, df["Lewiston"], label="Lewiston")
line3, = ax1.plot(df.index, df["South Portland"], label="South Portland")

ax1.set_xlabel("Date")
ax1.set_ylabel("Temperature (°F)")
ax1.set_title("Temperature Forecast with Predicted Hourly Load Overlay")
ax1.xaxis.set_major_locator(mdates.DayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax1.grid(True, which="major", axis="x")

for day in coldest_days:
    start = day
    end = start + pd.Timedelta(days=1)
    ax1.axvspan(start, end, alpha=0.15, color="blue")

for day in warmest_days:
    start = day
    end = start + pd.Timedelta(days=1)
    ax1.axvspan(start, end, alpha=0.15, color="red")

# Predicted load on right axis
ax1b = ax1.twinx()
line4, = ax1b.plot(
    forecast_df.index,
    forecast_df["Predicted_Load"],
    linestyle="--",
    linewidth=2,
    label="Predicted Hourly Load"
)
ax1b.set_ylabel("Predicted Load")

# Combined legend
lines = [line1, line2, line3, line4]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper left")

fig1.tight_layout()
st.pyplot(fig1)
