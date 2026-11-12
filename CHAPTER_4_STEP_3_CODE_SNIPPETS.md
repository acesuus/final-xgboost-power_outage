# Step 3. Feature Engineering: Code Snippets

## 1. Joining the daily weather forecast to each recloser-day

```python
panel = recloser_labels.merge(
    forecast_weather,
    on="date",
    how="left",
    validate="many_to_one"
)
```

## 2. Calendar features

```python
dates = pd.to_datetime(panel["date"])

panel["day_of_week"] = dates.dt.dayofweek
panel["is_weekend"] = dates.dt.dayofweek.ge(5).astype(int)

month_angle = 2 * np.pi * (dates.dt.month - 1) / 12
panel["month_sin"] = np.sin(month_angle)
panel["month_cos"] = np.cos(month_angle)

days_in_year = np.where(dates.dt.is_leap_year, 366, 365)
day_angle = 2 * np.pi * (dates.dt.dayofyear - 1) / days_in_year
panel["day_of_year_sin"] = np.sin(day_angle)
panel["day_of_year_cos"] = np.cos(day_angle)
```

## 3. Lagged and cumulative weather features

```python
weather = forecast_weather.sort_values("date").copy()

rain = weather["forecast_precipitation_mm"]
gust = weather["forecast_wind_gust_max_kmh"]

weather["forecast_precipitation_lag1d_mm"] = rain.shift(1)
weather["forecast_precipitation_sum_prev3d_mm"] = (
    rain.shift(1).rolling(3).sum()
)
weather["forecast_wind_gust_max_prev7d_kmh"] = (
    gust.shift(1).rolling(7).max()
)
```

## 4. Outage history features

```python
known = evidence[
    evidence["recloser"].eq(recloser)
    & evidence["date"].lt(target_date)
    & evidence["published_at_local"].le(prediction_cutoff)
]

recent = known[
    known["date"].ge(target_date - pd.Timedelta(days=7))
]

outage_days_prev7d = recent["date"].nunique()
```

## 5. Equipment-related features

```python
equipment_reports = known[known["equipment_fault_stated"].eq(1)]

equipment_fault_days_prev14d = equipment_reports.loc[
    equipment_reports["date"].ge(
        target_date - pd.Timedelta(days=14)
    ),
    "date"
].nunique()
```

## 6. Vegetation features and weather–vegetation interactions

```python
panel["vegetation_month"] = (
    pd.to_datetime(panel["date"]).dt.to_period("M") - 2
).astype(str)

panel = panel.merge(
    vegetation_by_recloser,
    on=["recloser", "vegetation_month"],
    how="left",
    validate="many_to_one"
)

panel["forecast_gust_x_veg5m"] = (
    panel["forecast_wind_gust_max_kmh"] * panel["veg_buffer_5m"]
)
panel["forecast_precipitation_x_veg5m"] = (
    panel["forecast_precipitation_mm"] * panel["veg_buffer_5m"]
)
```
