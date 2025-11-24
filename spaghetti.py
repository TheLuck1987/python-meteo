import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from plotly.colors import qualitative
import requests

# --- CONFIGURAZIONE POSIZIONI ---
LOCATIONS = {
    "home": {"lat": 45.7256, "lon": 12.6897},
    "cs": {"lat": 45.7585, "lon": 12.8447},
    "rugby": {"lat": 45.6956, "lon": 12.7102}
}

# --- COSTANTI ---
source_names = [
    "_gfs_global", "_ecmwf_ifs", "_ecmwf_ifs025", "_icon_global", "_icon_d2",
    "_meteofrance_arpege_europe", "_knmi_harmonie_arome_europe", "_dmi_harmonie_arome_europe",
    "_italia_meteo_arpae_icon_2i", "_bom_access_global", "_cma_grapes_global"
]
field_names = [
    "temperature_2m", "precipitation", "precipitation_probability", "cape",
    "wind_speed_10m", "wind_gusts_10m", "surface_pressure", "cloud_cover",
    "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "rain",
    "showers", "snowfall"
]
serie_names_it = [
    "Temperatura", "Precipitazioni", "Prob. Prec.", "CAPE", "Vento", "Raffiche",
    "Pressione", "Copertura Nuvolosa", "Umidità Relativa", "Punto di Rugiada",
    "Temp. Percepita", "Pioggia", "Rovesci", "Neve"
]
PAST_MEAN_LABEL = "MEDIA 50 ANNI"
X_AXIS_TITLE = "Ora (Locale)"

# --- FUNZIONI ---
def get_data(lat, lon, path):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,precipitation,wind_speed_10m"
    r = requests.get(url)
    r.raise_for_status()
    with open(path, "w") as f:
        json.dump(r.json(), f)

def robust_mean(arr):
    vals = np.array([v for v in arr if v is not None and not pd.isna(v)])
    if len(vals) == 0:
        return np.nan
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    filt = [v for v in vals if q1 - 1.5*iqr <= v <= q3 + 1.5*iqr]
    return np.mean(filt) if filt else np.median(vals)

def create_mid_graph(df, combined):
    fig = go.Figure()
    colors = qualitative.Light24
    for i, col in enumerate(combined.columns[1:]):
        fig.add_trace(go.Scatter(x=df["time"], y=combined[col], mode="lines",
                                 name=col, line=dict(color=colors[i % len(colors)], width=1.5)))
    tickvals = [t for t in df["time"] if t.hour == 0]
    fig.update_xaxes(tickvals=tickvals, tickformat="%a %d-%m")
    fig.update_layout(template="plotly_dark", hovermode="x unified", height=650, xaxis_title=X_AXIS_TITLE)
    return fig

def get_series(df, idx, combined_means, pdf_indexed=None):
    valid_fields = [f"{field_names[idx]}{src}" for src in source_names if f"{field_names[idx]}{src}" in df.columns and df[f"{field_names[idx]}{src}"].notna().any()]
    past_values = None
    if idx == 0 and pdf_indexed is not None:
        df['month_day_hour'] = df['time'].dt.strftime("%m-%d %H:%M")
        past_values = df['month_day_hour'].map(lambda x: robust_mean(pdf_indexed.get(x, [])))
        combined_means[PAST_MEAN_LABEL] = past_values
    if valid_fields:
        df["mean"] = df[valid_fields].apply(robust_mean, axis=1)
        combined_means[serie_names_it[idx]] = df["mean"]
    fig = go.Figure()
    if past_values is not None:
        fig.add_trace(go.Scatter(x=df["time"], y=past_values, mode="lines",
                                 name=PAST_MEAN_LABEL, line=dict(width=1.5, dash="longdash", color="red")))
    for f in valid_fields:
        fig.add_trace(go.Scatter(x=df["time"], y=df[f], mode="lines", name=f.replace(field_names[idx], ""), line=dict(width=1.5)))
    if valid_fields:
        fig.add_trace(go.Scatter(x=df["time"], y=df["mean"], mode="lines",
                                 name=serie_names_it[idx], line=dict(width=3, color="blue")))
    return fig

def create_final_html(content, title):
    timestamp_html = f'<div style="position:fixed;bottom:5px;right:10px;font-size:0.7em;color:#999;">Ultimo aggiornamento: {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}</div>'
    return f"<!DOCTYPE html><html lang='it'><head><meta charset='UTF-8'><title>{title}</title></head><body>{content}{timestamp_html}</body></html>"

# --- GENERAZIONE MAIN PAGE ---
def generate_forecast_pages():
    os.makedirs("output", exist_ok=True)

    # Carica storico
    with open("historical.json", "r") as pf:
        pdf = pd.DataFrame(json.load(pf)["hourly"])
    pdf["time"] = pd.to_datetime(pdf["time"], utc=True).dt.tz_convert("Europe/Rome")
    pdf['month_day_hour'] = pdf['time'].dt.strftime("%m-%d %H:%M")
    pdf_indexed = pdf.groupby('month_day_hour')['temperature_2m'].apply(list).to_dict()

    for folder, cfg in LOCATIONS.items():
        json_path = os.path.join(folder, "open-meteo.json")
        os.makedirs(folder, exist_ok=True)
        get_data(cfg["lat"], cfg["lon"], json_path)

        with open(json_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Rome")

        combined_means = {}
        figs = [get_series(df, i, combined_means, pdf_indexed if i == 0 else None) for i in range(len(field_names))]

        combined_df = pd.DataFrame({"time": df["time"]})
        for k,v in combined_means.items():
            combined_df[k] = v

        html_content = f"<h1>Previsioni Medie 7 Giorni - {folder.upper()}</h1>"
        html_content += pio.to_html(create_mid_graph(df, combined_df), full_html=False, include_plotlyjs='cdn')
        html_content += f"<h1>Previsioni Dettagliate</h1>"
        for fig in figs:
            html_content += pio.to_html(fig, full_html=False, include_plotlyjs=False)

        with open(f"output/index_{folder}.html", "w", encoding="utf-8") as f:
            f.write(create_final_html(html_content, f"Meteo Forecast - {folder.upper()}"))

if __name__ == "__main__":
    generate_forecast_pages()
