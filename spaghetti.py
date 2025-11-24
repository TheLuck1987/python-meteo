import json
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import requests

from datetime import datetime
from plotly.colors import qualitative
from scipy.stats import trim_mean

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

source_names = [ "_gfs_global", "_ecmwf_ifs", "_ecmwf_ifs025", "_icon_global", "_icon_d2",
                 "_meteofrance_arpege_europe", "_knmi_harmonie_arome_europe",
                 "_dmi_harmonie_arome_europe", "_italia_meteo_arpae_icon_2i",
                 "_bom_access_global", "_cma_grapes_global" ]

friendly_sources = [ "GFS", "ECMWF", "ECMWF_025", "ICON", "ICON_D2",
                     "ARPEGE", "KNMI", "DMI", "ARPAE", "BOM", "CMA" ]

field_names = [ "temperature_2m", "precipitation", "precipitation_probability",
                "cape", "wind_speed_10m", "wind_gusts_10m", "surface_pressure",
                "cloud_cover", "relative_humidity_2m", "dew_point_2m",
                "apparent_temperature", "rain", "showers", "snowfall" ]

serie_names = [ "T", "PREC", "PREC. PROB.", "CAPE", "WIND", "GUSTS", "PRESSURE",
                "CLOUD COVER", "HUMIDITY", "DEWPOINT", "APPARENT T", "RAIN",
                "SHOWERS", "SNOW" ]

titles = ["Temperature [°C]", "Precipitation [mm/h]", "Precipitation probablility [%]",
          "Cape index [J/kg]", "Wind speed [km/h]", "Wind gusts [km/h]",
          "Pressure [hPa]", "Cloud cover [%]", "Relative humidity [%]",
          "Dewpoint [°C]", "Apparent temperature [°C]", "Rain [mm/h]",
          "Showers [mm/h]", "Snow [cm/h]" ]

combined_means = {}

# -------------------------------------------------------------------
# LOAD HISTORICAL ONLY ONCE
# -------------------------------------------------------------------

with open("historical.json", "r", encoding="utf_8") as pf:
    past_data = json.load(pf)

HIST_PDF = pd.DataFrame(past_data["hourly"])
HIST_PDF["time"] = pd.to_datetime(HIST_PDF["time"], utc=True).dt.tz_convert("Europe/Rome")

# Create day+hour key for fast lookup
HIST_PDF["key"] = HIST_PDF["time"].dt.strftime("%m-%d %H")

# Dict: "MM-DD HH" → list of 50 years of values
HIST_CACHE = HIST_PDF.groupby("key")["temperature_2m"].apply(list).to_dict()

# -------------------------------------------------------------------
# OPTIMIZED robust_mean
# -------------------------------------------------------------------

def robust_mean(vals):
    vals = np.asarray(vals)
    vals = vals[~pd.isnull(vals)]
    if vals.size == 0:
        return np.nan

    q1 = np.percentile(vals, 25)
    q3 = np.percentile(vals, 75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (vals >= lower) & (vals <= upper)
    filtered = vals[mask]

    if filtered.size == 0:
        return np.median(vals)

    return filtered.mean()

# -------------------------------------------------------------------
# FAST historical lookup using pre-indexed dictionary
# -------------------------------------------------------------------

def get_past(date):
    key = date.strftime("%m-%d %H")
    values = HIST_CACHE.get(key, [])
    return robust_mean(values) if values else np.nan

# -------------------------------------------------------------------
# FETCH ACTUAL DATA
# -------------------------------------------------------------------

def get_data():
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=45.7256&longitude=12.6897&daily=wind_direction_10m_dominant"
        "&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,"
        "precipitation_probability,precipitation,cloud_cover,wind_speed_10m,wind_gusts_10m,"
        "wind_direction_10m,surface_pressure,cape,snowfall,showers,rain"
        "&models=ecmwf_ifs025,icon_d2,gfs_global,ecmwf_ifs,icon_global,gem_regional,"
        "knmi_harmonie_arome_europe,dmi_harmonie_arome_europe,meteofrance_arpege_europe,"
        "italia_meteo_arpae_icon_2i,bom_access_global,cma_grapes_global"
    )

    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# -------------------------------------------------------------------
# MID GRAPH (unchanged logic)
# -------------------------------------------------------------------

def create_mid_graph(df):
    fig = go.Figure()
    colors = qualitative.Light24

    def is_visible(idx):
        return True if idx in (0, 1, 2, 11) else "legendonly"

    def is_dashed(idx):
        return "longdash" if idx == 0 else "solid"

    for idx, col in enumerate(df.columns[1:]):
        fig.add_trace(go.Scatter(
            x=df["time"], y=df[col],
            mode="lines", name=col,
            line=dict(width=1.5, color=colors[idx % len(colors)], dash=is_dashed(idx)),
            visible=is_visible(idx)
        ))

    fig.update_layout(
        xaxis_title="Time (Local)",
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.35,
            xanchor="center", x=0.5,
        ),
        hovermode="x unified",
        height=650
    )

    tickvals = [t for t in df["time"] if t.hour == 0]

    fig.update_xaxes(
        ticklabelmode="period",
        minor=dict(ticks="inside", showgrid=True, dtick=3600000,
                   tick0=df["time"].min(), griddash='dot',
                   gridcolor='rgba(50,50,50,0.5)'),
        showgrid=True,
        dtick=3600000,
        gridcolor="rgba(255,255,255,0.5)",
        tickvals=tickvals,
        tickformat="%a %d-%m",
        hoverformat="%a %d-%m %H:%M"
    )

    for t in df["time"]:
        if t.hour == 0:
            fig.add_vline(x=t, line_width=2, color="white")
        elif t.hour == 12:
            fig.add_vline(x=t, line_width=1.5, color="rgba(130,130,130,0.5)")

    return fig

# -------------------------------------------------------------------
# SERIES GRAPH (no more heavy calculations)
# -------------------------------------------------------------------

def get_series(df, serie_index):
    fig = go.Figure()

    model_fields = [
        field_names[serie_index] + src
        for src in source_names
        if (field_names[serie_index] + src) in df.columns
    ]

    # Add historical only for temperature
    if serie_index == 0:
        past_values = [get_past(t) for t in df["time"]]
        fig.add_trace(go.Scatter(
            x=df["time"], y=past_values,
            mode="lines", name="PAST 50y",
            line=dict(width=1.5, dash="longdash", color="red")
        ))

    # Add individual model lines
    for mf in model_fields:
        idx = source_names.index(mf.replace(field_names[serie_index], ""))
        fig.add_trace(go.Scatter(
            x=df["time"], y=df[mf],
            mode="lines",
            name=friendly_sources[idx],
            line=dict(width=1.5)
        ))

    # Add precomputed average
    mean_key = serie_names[serie_index]
    if mean_key in combined_means:
        fig.add_trace(go.Scatter(
            x=df["time"], y=combined_means[mean_key],
            mode="lines",
            name=mean_key,
            line=dict(width=3, color="blue")
        ))

    fig.update_layout(
        title=titles[serie_index],
        xaxis_title="Time (Local)",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="top", y=-0.35, xanchor="center", x=0.5),
        hovermode="x unified",
        height=400
    )

    tickvals = [t for t in df["time"] if t.hour == 0]

    fig.update_xaxes(
        ticklabelmode="period",
        minor=dict(ticks="inside", showgrid=True, dtick=3600000,
                   tick0=df["time"].min(), griddash='dot',
                   gridcolor='rgba(50,50,50,0.5)'),
        showgrid=True,
        dtick=3600000,
        gridcolor="rgba(255,255,255,0.5)",
        tickvals=tickvals,
        tickformat="%a %d-%m",
        hoverformat="%a %d-%m %H:%M"
    )

    return fig

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

data = get_data()
df = pd.DataFrame(data["hourly"])
df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Rome")

# Remove past hours, ensure copy
df = df[df["time"] >= df.iloc[0]["time"].normalize() + pd.Timedelta(hours=datetime.now().hour)].copy()

# Precompute means for all variables in ONE pass
for i, field in enumerate(field_names):
    cols = [field + src for src in source_names if (field + src) in df.columns]
    if cols:
        combined_means[serie_names[i]] = df[cols].apply(robust_mean, axis=1).to_numpy()

# Build combined_df for mid graph
combined_df = pd.DataFrame({"time": df["time"]})
for k, v in combined_means.items():
    combined_df[k] = v

# Generate full HTML
html = "<h1>Next 7 days overview</h1>"
html += pio.to_html(create_mid_graph(combined_df), full_html=False, include_plotlyjs='cdn')

html += "<h1 style='padding-top: 30px;'>Detailed Model Forecasts</h1>"
for i in range(len(field_names)):
    html += pio.to_html(get_series(df, i), full_html=False, include_plotlyjs=False)

# final output
final_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Meteo Forecast Analysis</title>
<style>
    body {{ background-color:#111; color:#fff; font-family: 'Inter', sans-serif; }}
    h1 {{ color:#4CAF50; border-bottom:2px solid #4CAF50; padding:10px; }}
</style>
</head>
<body>
{html}
</body>
</html>
"""

print(final_html)
