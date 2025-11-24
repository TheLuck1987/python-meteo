import json
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import requests
import os

from datetime import datetime
from plotly.colors import qualitative
from plotly.subplots import make_subplots

# --- CARICAMENTO UNA SOLA VOLTA DELLO STORICO ---
with open("historical.json", "r", encoding="utf_8") as pf:
    _past_data = json.load(pf)

HISTORICAL_PDF = pd.DataFrame(_past_data["hourly"])
HISTORICAL_PDF["time"] = pd.to_datetime(HISTORICAL_PDF["time"], utc=True).dt.tz_convert("Europe/Rome")

# --- CONFIGURAZIONE POSIZIONI ---

LOCATIONS = {
    "home": {"lat": 45.7256, "lon": 12.6897},
    "cs": {"lat": 45.7585, "lon": 12.8447},
    "rugby": {"lat": 45.6956, "lon": 12.7102}
}

# --- COSTANTI E TRADUZIONI ---

source_names = [
    "_gfs_global", "_ecmwf_ifs", "_ecmwf_ifs025", "_icon_global", "_icon_d2",
    "_meteofrance_arpege_europe", "_knmi_harmonie_arome_europe",
    "_dmi_harmonie_arome_europe", "_italia_meteo_arpae_icon_2i",
    "_bom_access_global", "_cma_grapes_global"
]

friendly_sources = [
    "GFS", "ECMWF", "ECMWF_025", "ICON", "ICON_D2",
    "ARPEGE", "KNMI", "DMI", "ARPAE", "BOM", "CMA"
]

field_names = [
    "temperature_2m", "precipitation", "precipitation_probability", "cape",
    "wind_speed_10m", "wind_gusts_10m", "surface_pressure", "cloud_cover",
    "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "rain", "showers", "snowfall"
]

serie_names_it = [
    "Temperatura", "Precipitazioni", "Prob. Prec.", "CAPE", "Vento", "Raffiche",
    "Pressione", "Copertura Nuvolosa", "Umidità Relativa", "Punto di Rugiada",
    "Temp. Percepita", "Pioggia", "Rovesci", "Neve"
]

titles_it = [
    "Temperatura [°C]", "Precipitazioni [mm/h]", "Probabilità di Precipitazioni [%]",
    "Indice CAPE [J/kg]", "Velocità del Vento [km/h]", "Raffiche di Vento [km/h]",
    "Pressione [hPa]", "Copertura Nuvolosa [%]", "Umidità Relativa [%]",
    "Punto di Rugiada [°C]", "Temperatura Percepita [°C]", "Pioggia [mm/h]",
    "Rovesci [mm/h]", "Neve [cm/h]"
]

DAY_NAMES_IT = {
    'Monday': 'Lunedì', 'Tuesday': 'Martedì', 'Wednesday': 'Mercoledì',
    'Thursday': 'Giovedì', 'Friday': 'Venerdì', 'Saturday': 'Sabato',
    'Sunday': 'Domenica'
}

X_AXIS_TITLE = "Ora (Locale)"
TITLE_MAIN = "Previsioni Medie 7 Giorni"
TITLE_DETAIL = "Previsioni Dettagliate per Modello"
PAST_MEAN_LABEL = "MEDIA 50 ANNI"

# --- FUNZIONI DI BASE ---

def get_data(latitude, longitude, local_path):
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}"
        f"&daily=wind_direction_10m_dominant"
        f"&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,"
        f"precipitation_probability,precipitation,cloud_cover,wind_speed_10m,wind_gusts_10m,"
        f"wind_direction_10m,surface_pressure,cape,snowfall,showers,rain"
        f"&models=ecmwf_ifs025,icon_d2,gfs_global,ecmwf_ifs,icon_global,gem_regional,"
        f"knmi_harmonie_arome_europe,dmi_harmonie_arome_europe,meteofrance_arpege_europe,"
        f"italia_meteo_arpae_icon_2i,bom_access_global,cma_grapes_global"
    )
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def robust_mean(row):
    if isinstance(row, list) or isinstance(row, np.ndarray):
        vals = np.array(row)
    else:
        vals = row.dropna().values
    if len(vals) == 0:
        return np.nan
    Q1, Q3 = np.percentile(vals, [25, 75])
    IQR = Q3 - Q1
    low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    filt = [v for v in vals if low <= v <= high]
    return np.mean(filt) if len(filt) else np.median(vals)

def get_past(date, pdf):
    values = []
    for y in range(1974, 2025):
        d2 = date.replace(year=y)
        match = pdf.loc[pdf["time"] == d2, "temperature_2m"]
        if not match.empty:
            values.append(match.iloc[0])
    return robust_mean(values)

# --- GENERAZIONE GRAFICI ---

def create_mid_graph(df, combined_data):
    fig = go.Figure()
    colors = qualitative.Light24

    def is_visible(idx):
        return True if idx in (0, 1, 2, 11) else "legendonly"

    def is_dashed(idx):
        return "longdash" if idx == 0 else "solid"

    for idx, col in enumerate(combined_data.columns[1:]):
        fig.add_trace(go.Scatter(
            x=df["time"], y=combined_data[col],
            mode="lines", name=col,
            line=dict(width=1.5, color=colors[idx % len(colors)], dash=is_dashed(idx)),
            visible=is_visible(idx)
        ))

    fig.update_layout(
        xaxis_title=X_AXIS_TITLE,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="top", y=-0.35, xanchor="center", x=0.5),
        hovermode="x unified",
        height=650
    )

    tickvals = [t for t in df["time"] if t.hour == 0]
    fig.update_xaxes(
        ticklabelmode="period",
        minor=dict(ticks="inside", showgrid=True, dtick=3600000, tick0=df["time"].min(),
                   griddash='dot', gridcolor='rgba(50,50,50,0.5)'),
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

def get_series(df, serie_index, current_combined_means):

    def valid_field_names(index):
        temp = [field_names[index] + s for s in source_names]
        return [f for f in temp if f in df.columns and df[f].notna().any()]

    def rename_serie(field_index, field):
        idx = source_names.index(field.replace(field_names[field_index], ""))
        return friendly_sources[idx]

    valid_fields = valid_field_names(serie_index)
    past_values = None

    # --- QUI USIAMO LO STORICO CARICATO UNA SOLA VOLTA ---
    if serie_index == 0:
        pdf = HISTORICAL_PDF
        past_values = [get_past(d, pdf) for d in df["time"]]
        current_combined_means[PAST_MEAN_LABEL] = past_values.copy()

    if valid_fields:
        df["mean"] = df[valid_fields].apply(robust_mean, axis=1)
        current_combined_means[serie_names_it[serie_index]] = df["mean"].copy()

    fig = go.Figure()

    if past_values is not None:
        df["past"] = past_values
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["past"],
            mode="lines", name=PAST_MEAN_LABEL,
            line=dict(width=1.5, dash="longdash", color="red"),
            visible=True
        ))

    for f in valid_fields:
        fig.add_trace(go.Scatter(
            x=df["time"], y=df[f],
            mode="lines", name=rename_serie(serie_index, f),
            line=dict(width=1.5), visible=True
        ))

    if valid_fields:
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["mean"],
            mode="lines", name=serie_names_it[serie_index],
            line=dict(width=3, dash="solid", color="blue"), visible=True
        ))

    fig.update_layout(
        title=titles_it[serie_index],
        xaxis_title=X_AXIS_TITLE,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="top", y=-0.35,
                    xanchor="center", x=0.5),
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

    for t in df["time"]:
        if t.hour == 0:
            fig.add_vline(x=t, line_width=2, line_dash="solid", line_color="white")
        elif t.hour == 12:
            fig.add_vline(x=t, line_width=1.5, line_dash="solid",
                          line_color="rgba(130,130,130,0.5)")

    return fig

# --- HTML TEMPLATE ---

def create_final_html(content, title, timestamp_html):
    return f"""
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script>
        function refreshPage() {{
            window.location.reload(true);
            setPageRefresh();
        }}
        function setPageRefresh() {{
            let now = new Date();
            let currentSeconds = now.getMinutes() * 60 + now.getSeconds() + now.getMilliseconds() / 1000;
            let nextTargetSeconds = currentSeconds < 60 ? 60 : currentSeconds < 1860 ? 1860 : 3660;
            let diff = nextTargetSeconds - currentSeconds;
            if (diff < 0) diff += 3600;
            setTimeout(refreshPage, Math.round(diff * 1000));
        }}
    </script>
    <style>
        body {{ background-color: #111; color: #fff; font-family: 'Inter'; margin: 0; }}
        h1 {{ color: #4CAF50; border-bottom: 2px solid #4CAF50; padding: 10px; }}
        h2 {{ color: #777; border-bottom: 1px solid #777; padding: 10px; }}
        .day-link {{ color: #4CAF50; margin-right: 20px; font-size: 1.1em; text-decoration: none; }}
        .day-link:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
{content}
{timestamp_html}
<script>setPageRefresh()</script>
</body>
</html>
"""

# --- PROCESSO PRINCIPALE ---

def generate_forecast_pages(folder_name, latitude, longitude):
    print(f"--- Elaborazione iniziata: {folder_name} ---")

    os.makedirs(folder_name, exist_ok=True)
    json_path = os.path.join(folder_name, "open-meteo.json")
    get_data(latitude, longitude, json_path)

    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    timestamp_html = (
        f'<div style="position: fixed; bottom: 5px; right: 10px; '
        f'font-size: 0.7em; color: #999; z-index: 1000;">Ultimo aggiornamento: {now}</div>'
    )

    with open(json_path, "r", encoding="utf_8") as f:
        data = json.load(f)

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Rome")
    df = df[df["time"] >= df.iloc[0]["time"].normalize() + pd.Timedelta(hours=datetime.now().hour)]
    unique_days = df["time"].dt.date.unique()

    combined_means_main = {}
    figs_combined = [get_series(df, i, combined_means_main) for i in range(len(field_names))]

    combined_df_main = pd.DataFrame({"time": df["time"]})
    for k, v in combined_means_main.items():
        combined_df_main[k] = v

    link_html = "<h1 style='padding-top:30px;'>Navigazione Giornaliera</h1><div style='padding:10px;'>"
    for idx, day in enumerate(unique_days):
        dn_en = day.strftime("%A")
        dn_it = DAY_NAMES_IT.get(dn_en, dn_en)
        link_html += f"<a class='day-link' href='{idx}.html'>{dn_it} {day.strftime('%d-%m')}</a>"
    link_html += "</div>"

    html_main = f"<h1>{TITLE_MAIN} - {folder_name.upper()}</h1>" + link_html
    html_main += pio.to_html(create_mid_graph(df, combined_df_main), full_html=False, include_plotlyjs='cdn')
    html_main += f"<h1 style='padding-top:30px;'>{TITLE_DETAIL}</h1>"

    for fig in figs_combined:
        html_main += pio.to_html(fig, full_html=False, include_plotlyjs=False)

    with open(os.path.join(folder_name, "index.html"), "w", encoding="utf-8") as f:
        f.write(create_final_html(html_main, f"Meteo Forecast - {folder_name.upper()}", timestamp_html))

    # PAGINE GIORNALIERE
    for idx, day in enumerate(unique_days):
        df_day = df[df["time"].dt.date == day]

        combined_means_daily = {}
        figs_daily = [get_series(df_day, i, combined_means_daily) for i in range(len(field_names))]

        cdf = pd.DataFrame({"time": df_day["time"]})
        for k, v in combined_means_daily.items():
            cdf[k] = v

        dn_en = day.strftime("%A")
        dn_it = DAY_NAMES_IT.get(dn_en, dn_en)
        label = f"{dn_it} {day.strftime('%d-%m')}"

        html_day = (
            f"<h1>Previsioni Dettagliate per {label} ({folder_name.upper()})</h1>"
            f"<p style='padding:10px;'><a href='index.html' class='day-link'>← Torna alla panoramica</a></p>"
        )

        html_day += pio.to_html(create_mid_graph(df_day, cdf), full_html=False, include_plotlyjs='cdn')
        html_day += f"<h2 style='padding-top:30px;'>Dettagli Modelli</h2>"

        for fig in figs_daily:
            html_day += pio.to_html(fig, full_html=False, include_plotlyjs=False)

        with open(os.path.join(folder_name, f"{idx}.html"), "w", encoding="utf-8") as f:
            f.write(create_final_html(html_day, f"Meteo Forecast - {folder_name.upper()}", timestamp_html))

    print(f"--- Completato: {folder_name} ---\n")

# --- ESECUZIONE GLOBALE ---

if __name__ == "__main__":
    for folder, config in LOCATIONS.items():
        generate_forecast_pages(folder, config["lat"], config["lon"])
