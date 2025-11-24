import json
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import requests

from datetime import datetime
from plotly.colors import qualitative
from plotly.subplots import make_subplots
from scipy.stats import trim_mean

source_names = [ "_gfs_global", "_ecmwf_ifs", "_ecmwf_ifs025", "_icon_global", "_icon_d2", "_meteofrance_arpege_europe", "_knmi_harmonie_arome_europe", "_dmi_harmonie_arome_europe", "_italia_meteo_arpae_icon_2i", "_bom_access_global", "_cma_grapes_global" ]
friendly_sources = [ "GFS", "ECMWF", "ECMWF_025", "ICON", "ICON_D2", "ARPEGE", "KNMI", "DMI", "ARPAE", "BOM", "CMA" ]
field_names = [ "temperature_2m", "precipitation", "precipitation_probability", "cape", "wind_speed_10m", "wind_gusts_10m", "surface_pressure", "cloud_cover", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "rain", "showers", "snowfall"  ]
serie_names = [ "T", "PREC", "PREC. PROB.", "CAPE", "WIND", "GUSTS", "PRESSURE", "CLOUD COVER", "HUMIDITY", "DEWPOINT", "APPARENT T", "RAIN", "SHOWERS", "SNOW" ]
titles = ["Temperature [°C]", "Precipitation [mm/h]", "Precipitation probablility [%]", "Cape index [J/kg]", "Wind speed [km/h]", "Wind gusts [km/h]", "Pressure [hPa]", "Cloud cover [%]", "Relative humidity [%]", "Dewpoint [°C]", "Apparent temperature [°C]", "Rain [mm/h]", "Showers [mm/h]", "Snow [cm/h]" ]

combined_means = {}

def get_data():
    url = "https://api.open-meteo.com/v1/forecast?latitude=45.7256&longitude=12.6897&daily=wind_direction_10m_dominant&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,precipitation,cloud_cover,wind_speed_10m,wind_gusts_10m,wind_direction_10m,surface_pressure,cape,snowfall,showers,rain&models=ecmwf_ifs025,icon_d2,gfs_global,ecmwf_ifs,icon_global,gem_regional,knmi_harmonie_arome_europe,dmi_harmonie_arome_europe,meteofrance_arpege_europe,italia_meteo_arpae_icon_2i,bom_access_global,cma_grapes_global"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()   # <--- QUI: ritorna i dati, non salva su file

def robust_mean(row):
    if isinstance(row, list) or isinstance(row, np.ndarray):
        vals = np.array(row)
    else:
        vals = row.dropna().values
    if len(vals) == 0:
        return np.nan        
    Q1 = np.percentile(vals, 25)
    Q3 = np.percentile(vals, 75)
    IQR = Q3 - Q1        
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    filtered = [v for v in vals if lower_fence <= v <= upper_fence]
    if len(filtered) == 0:
        return np.median(vals)
    return np.mean(filtered)

def get_past(date, pdf):
    values = []
    for y in range(1974, 2025):
        var_date = date.replace(year=y)
        match = pdf.loc[pdf["time"] == var_date, "temperature_2m"]
        if not match.empty:
            values.append(match.iloc[0])
    return robust_mean(values)

def create_mid_graph(df):
    fig = go.Figure()
    colors = qualitative.Light24

    def is_visible(idx):
        return True if idx in (0,1,2,11) else "legendonly"

    def is_dashed(idx):
        return "longdash" if idx == 0 else "solid"

    for idx, col in enumerate(combined_df.columns[1:]):
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
        minor=dict(ticks="inside", showgrid=True, dtick=60*60*1000, 
                   tick0=df["time"].min(), griddash='dot', gridcolor='rgba(50,50,50,0.5)'),
        showgrid=True,
        dtick=3600000,
        gridcolor="rgba(255,255,255,0.5)",
        tickvals=tickvals,
        tickformat="%a %d-%m",
        hoverformat="%a %d-%m %H:%M"
    )

    for t in df["time"]:
        if t.hour == 0:
            fig.add_vline(x=t, line_width=2, line_dash="solid", line_color="white")
        elif t.hour == 12:
            fig.add_vline(x=t, line_width=1.5, line_dash="solid",
                          line_color="rgba(130,130,130,0.5)")

    return fig

def get_series(df, serie_index):
    def valid_field_names(index):
        temp_fields = [field_names[index] + s for s in source_names]
        return [f for f in temp_fields if f in df.columns and df[f].notna().any()]

    def rename_serie(field_index, field):
        index = source_names.index(field.replace(field_names[field_index], ""))
        return friendly_sources[index]

    valid_fields = valid_field_names(serie_index)

    past_values = None
    if serie_index == 0:
        # Carica storico dal file (se lo vuoi solo in memoria, modifica anche questo)
        with open("D:\\Meteo\\py\\historical.json", "r", encoding="utf_8") as pf:
            past_data = json.load(pf)
        pdf = pd.DataFrame(past_data["hourly"])
        pdf["time"] = pd.to_datetime(pdf["time"], utc=True).dt.tz_convert("Europe/Rome")

        past_values = [get_past(d, pdf) for d in df["time"]]
        combined_means["PAST 50y"] = past_values.copy()

    if valid_fields:
        df["mean"] = df[valid_fields].apply(robust_mean, axis=1)
        combined_means[serie_names[serie_index]] = df["mean"].copy()

    fig = go.Figure()

    if past_values is not None:
        df["past"] = past_values
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["past"],
            mode="lines", name="PAST 50y",
            line=dict(width=1.5, dash="longdash", color="red"),
        ))

    for field in valid_fields:
        fig.add_trace(go.Scatter(
            x=df["time"], y=df[field],
            mode="lines", name=rename_serie(serie_index, field),
            line=dict(width=1.5)
        ))

    if valid_fields:
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["mean"],
            mode="lines", name=serie_names[serie_index],
            line=dict(width=3, dash="solid", color="blue")
        ))

    fig.update_layout(
        title=titles[serie_index],
        xaxis_title="Time (Local)",
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.35,
            xanchor="center", x=0.5
        ),
        hovermode="x unified",
        height=400
    )

    tickvals = [t for t in df["time"] if t.hour == 0]
    fig.update_xaxes(
        ticklabelmode="period",
        minor=dict(ticks="inside", showgrid=True, dtick=60*60*1000,
                   tick0=df["time"].min(), griddash='dot', gridcolor='rgba(50,50,50,0.5)'),
        showgrid=True,
        dtick=3600000,
        gridcolor="rgba(255,255,255,0.5)",
        tickvals=tickvals,
        tickformat="%a %d-%m",
        hoverformat="%a %d-%m %H:%M"
    )

    for t in df["time"]:
        if t.hour == 0:
            fig.add_vline(x=t, line_width=2, line_dash="solid", line_color="white")
        elif t.hour == 12:
            fig.add_vline(x=t, line_width=1.5, line_dash="solid",
                          line_color="rgba(130,130,130,0.5)")

    return fig


# ---- FLUSSO PRINCIPALE ----

data = get_data()

df = pd.DataFrame(data["hourly"])
df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Rome")

df = df[df['time'] >= df.iloc[0]['time'].normalize() + pd.Timedelta(hours=datetime.now().hour)]

figs = []
for i in range(len(field_names)):
    figs.append(get_series(df, i))

combined_df = pd.DataFrame({"time": df["time"]})
for k, v in combined_means.items():
    combined_df[k] = v

html_content = "<h1>Next 7 days overview</h1>"
combined_fig = create_mid_graph(combined_df)
html_content += pio.to_html(combined_fig, full_html=False, include_plotlyjs='cdn')

html_content += "<h1 style='padding-top: 30px;'>Detailed Model Forecasts</h1>"
for fig in figs:
    html_content += pio.to_html(fig, full_html=False, include_plotlyjs=False)

final_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Meteo Forecast Analysis</title>
    <script>
        function refreshPage() {{
            console.log('Eseguo il refresh della pagina...');
            window.location.reload(true);
            setPageRefresh();
        }}
        function setPageRefresh() {{
            let now = new Date();
            let currentSeconds = now.getMinutes() * 60 + now.getSeconds() + now.getMilliseconds() / 1000;
            let nextTargetSeconds = currentSeconds < 60 ? 60 : currentSeconds < 31*60 ? 31*60 : 60 + 3600;
            let difference = nextTargetSeconds - currentSeconds;
            if (difference < 0) difference += 3600;
            const delay = Math.round(difference * 1000);
            console.log(`Il prossimo refresh è programmato tra ${{delay / 1000}} secondi.`);
            setTimeout(refreshPage, delay);
        }}
    </script>
    <style>
        body {{ background-color: #111; color: #fff; font-family: 'Inter', sans-serif; padding: 0; margin: 0; overflow-x: hidden; }}
        h1 {{ color: #4CAF50; border-bottom: 2px solid #4CAF50; padding: 10px; }}
    </style>
</head>
<body>
{html_content}
<script>setPageRefresh()</script>
</body>
</html>
"""

# 👉 QUI LA MAGIA
def run_and_get_html():
    return final_html
