import json
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import requests

from datetime import datetime
from plotly.colors import qualitative
from plotly.subplots import make_subplots
# Ho rimosso l'importazione di trim_mean se non la usi nel codice
# from scipy.stats import trim_mean 

# --- COSTANTI E TRADUZIONI ---

source_names = [ "_gfs_global", "_ecmwf_ifs", "_ecmwf_ifs025", "_icon_global", "_icon_d2", "_meteofrance_arpege_europe", "_knmi_harmonie_arome_europe", "_dmi_harmonie_arome_europe", "_italia_meteo_arpae_icon_2i", "_bom_access_global", "_cma_grapes_global" ]
friendly_sources = [ "GFS", "ECMWF", "ECMWF_025", "ICON", "ICON_D2", "ARPEGE", "KNMI", "DMI", "ARPAE", "BOM", "CMA" ]
field_names = [ "temperature_2m", "precipitation", "precipitation_probability", "cape", "wind_speed_10m", "wind_gusts_10m", "surface_pressure", "cloud_cover", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "rain", "showers", "snowfall" ]

# TRADUZIONI IN ITALIANO
serie_names_it = [ "Temperatura", "Precipitazioni", "Prob. Prec.", "CAPE", "Vento", "Raffiche", "Pressione", "Copertura Nuvolosa", "Umidità Relativa", "Punto di Rugiada", "Temp. Percepita", "Pioggia", "Rovesci", "Neve" ]
titles_it = ["Temperatura [°C]", "Precipitazioni [mm/h]", "Probabilità di Precipitazioni [%]", "Indice CAPE [J/kg]", "Velocità del Vento [km/h]", "Raffiche di Vento [km/h]", "Pressione [hPa]", "Copertura Nuvolosa [%]", "Umidità Relativa [%]", "Punto di Rugiada [°C]", "Temperatura Percepita [°C]", "Pioggia [mm/h]", "Rovesci [mm/h]", "Neve [cm/h]" ]

# TITOLI E ASSI GENERALI
X_AXIS_TITLE = "Ora (Locale)"
TITLE_MAIN = "Previsioni Medie 7 Giorni"
TITLE_DETAIL = "Previsioni Dettagliate per Modello"
PAST_MEAN_LABEL = "MEDIA 50 ANNI"


# --- FUNZIONI DI BASE ---

def get_data():
    url = "https://api.open-meteo.com/v1/forecast?latitude=45.7256&longitude=12.6897&daily=wind_direction_10m_dominant&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,precipitation,cloud_cover,wind_speed_10m,wind_gusts_10m,wind_direction_10m,surface_pressure,cape,snowfall,showers,rain&models=ecmwf_ifs025,icon_d2,gfs_global,ecmwf_ifs,icon_global,gem_regional,knmi_harmonie_arome_europe,dmi_harmonie_arome_europe,meteofrance_arpege_europe,italia_meteo_arpae_icon_2i,bom_access_global,cma_grapes_global"
    local_path = "open-meteo.json"
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
        var_date = date.replace(year = y)
        match = pdf.loc[pdf["time"] == var_date, "temperature_2m"]
        if not match.empty:
            values.append(match.iloc[0])
    return robust_mean(values)

# --- FUNZIONI DI GENERAZIONE GRAFICI E HTML ---

def create_mid_graph(df, combined_data): # ACCETTA I DATI COMBINATI LOCALI
    fig = go.Figure()
    colors = qualitative.Light24
    def is_visible(idx):
        if idx == 0 or idx == 1 or idx == 2 or idx == 11:
            return True
        else:
            return "legendonly"
    def is_dashed(idx):
        if idx == 0:
            return "longdash"
        else:
            return "solid"
            
    # Usa i dati combinati passati (combined_data)
    for idx, col in enumerate(combined_data.columns[1:]): 
        fig.add_trace(go.Scatter(
            x=df["time"], y=combined_data[col],
            mode="lines", name=col,
            line=dict(width=1.5,color=colors[idx % len(colors)],dash=is_dashed(idx)),
            visible=is_visible(idx) 
        ))
        
    fig.update_layout(
        xaxis_title=X_AXIS_TITLE,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.35,
            xanchor="center", x=0.5,
            traceorder="normal"
        ),
        hovermode="x unified",
        height=650
    )    
    
    # ... (omissis: logica assi X e linee verticali)
    tickvals = [t for t in df["time"] if t.hour == 0]    
    fig.update_xaxes(
        ticklabelmode= "period", 
        minor=dict(ticks="inside", showgrid=True, dtick=60*60*1000, tick0=df["time"].min(), griddash='dot', gridcolor='rgba(50,50,50,0.5)'),
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
            fig.add_vline(x=t, line_width=1.5, line_dash="solid", line_color="rgba(130,130,130,0.5)")        
    return fig

def get_series(df, serie_index, current_combined_means): # ACCETTA E MODIFICA I DATI MEDI LOCALI 
    def valid_field_names(index):
        temp_fields = []
        for i in range(len(friendly_sources)):
            temp_fields.append(field_names[index] + source_names[i])
        return [f for f in temp_fields if f in df.columns and df[f].notna().any()]         
    
    def rename_serie(field_index, field) -> str:
        index = source_names.index(field.replace(field_names[field_index], ""))
        return friendly_sources[index]        
        
    valid_fields = valid_field_names(serie_index)    
    past_values = None
    
    if serie_index == 0:
        with open("historical.json", "r", encoding="utf_8") as pf:
            past_data = json.load(pf)
        pdf = pd.DataFrame(past_data["hourly"])
        pdf["time"] = pd.to_datetime(pdf["time"], utc=True).dt.tz_convert("Europe/Rome")
        past_values = []
        for d in df["time"]:
            past_values.append(get_past(d, pdf))
        current_combined_means[PAST_MEAN_LABEL] = past_values.copy() # USA VARIABILE LOCALE
        
    if valid_fields:
        df["mean"] = df[valid_fields].apply(robust_mean, axis=1)
        current_combined_means[serie_names_it[serie_index]] = df["mean"].copy() # USA VARIABILE LOCALE E NOME TRADOTTO
        
    fig = go.Figure()
    if past_values != None:
        df["past"] = past_values
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["past"],
            mode="lines", name=PAST_MEAN_LABEL,
            line=dict(width=1.5, dash="longdash", color="red"),
            visible=True 
        ))            
        
    for i in range(len(valid_fields)):
        fig.add_trace(go.Scatter(
            x=df["time"], y=df[valid_fields[i]],
            mode="lines", name=rename_serie(serie_index, valid_fields[i]),
            line=dict(width=1.5),
            visible=True 
        ))        
        
    if valid_fields:
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["mean"],
            mode="lines", name=serie_names_it[serie_index], # USA NOME TRADOTTO
            line=dict(width=3, dash="solid", color="blue"),
            visible=True 
        ))     
        
    fig.update_layout(
        title=titles_it[serie_index], # USA TITOLO TRADOTTO
        xaxis_title=X_AXIS_TITLE,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.35,
            xanchor="center", x=0.5,
            traceorder="normal"
        ),
        hovermode="x unified",
        height=400
    )    
    
    # ... (omissis: logica assi X e linee verticali)
    tickvals = [t for t in df["time"] if t.hour == 0]    
    fig.update_xaxes(
        ticklabelmode= "period", 
        minor=dict(ticks="inside", showgrid=True, dtick=60*60*1000, tick0=df["time"].min(), griddash='dot', gridcolor='rgba(50,50,50,0.5)'),
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
            fig.add_vline(x=t, line_width=1.5, line_dash="solid", line_color="rgba(130,130,130,0.5)")  
    return fig

def create_final_html(content, title):
    return f"""
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script>
        // Logica di refresh (mantenuta)
        function refreshPage() {{
            console.log('Eseguo il refresh della pagina...');
            window.location.reload(true);
            setPageRefresh();
        }}
        function setPageRefresh() {{
            let now = new Date();
            let currentSeconds = now.getMinutes() * 60 + now.getSeconds() + now.getMilliseconds() / 1000;
            // Target seconds: 1 min, 31 min, or 1 hr + 1 min (per il loop orario)
            let nextTargetSeconds = currentSeconds < 1 * 60 ? 1 * 60 : currentSeconds < 31 * 60 ? 31 * 60 : 1 * 60 + 3600;
            let difference = nextTargetSeconds - currentSeconds;
            if (difference < 0)
                difference += 3600;
            const delay = Math.round(difference * 1000);
            console.log(`Il prossimo refresh è programmato tra ${{delay / 1000}} secondi.`);
            setTimeout(refreshPage, delay);
        }}
    </script>
    <style>
        body {{ background-color: #111; color: #fff; font-family: 'Inter', sans-serif; padding: 0px; margin: 0px; max-width: calc(100% - 20px); overflow-x: hidden; }}
        h1 {{ color: #4CAF50; border-bottom: 2px solid #4CAF50; max-width: calc(100% - 20px); overflow: hidden; padding: 10px; }}
        h2 {{ color: #777; border-bottom: 1px solid #777; max-width: calc(100% - 20px); overflow: hidden; padding: 10px; }}
    </style>
</head>
<body>
{content}
<script>setPageRefresh()</script>
</body>
</html>
"""

# --- ESECUZIONE PRINCIPALE ---

get_data()

with open("open-meteo.json", "r", encoding="utf_8") as f:
    data = json.load(f)    
df = pd.DataFrame(data["hourly"])
df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Rome")
df = df[df['time'] >= df.iloc[0]['time'].normalize() + pd.Timedelta(hours=datetime.now().hour)]

# Ottieni la lista dei giorni unici nel DataFrame
unique_days = df["time"].dt.date.unique()

# --- GENERAZIONE PAGINA PRINCIPALE (index.html) ---

combined_means_main = {}
figs_combined = []

for i in range(len(field_names)):
    # Raccoglie i dati per il grafico combinato generale
    figs_combined.append(get_series(df, i, combined_means_main)) 

# Calcola il DataFrame combinato per la pagina principale
combined_df_main = pd.DataFrame({"time": df["time"]})
for k, v in combined_means_main.items():
    combined_df_main[k] = v

html_content_main = f"<h1>{TITLE_MAIN}</h1>"
combined_fig = create_mid_graph(df, combined_df_main)
html_content_main += pio.to_html(combined_fig, full_html=False, include_plotlyjs='cdn')
html_content_main += f"<h1 style='padding-top: 30px;'>{TITLE_DETAIL}</h1>"
for i in range(len(figs_combined)):
    html_content_main += pio.to_html(figs_combined[i], full_html=False, include_plotlyjs=False)

# SALVATAGGIO PAGINA PRINCIPALE
final_html_main = create_final_html(html_content_main, "Meteo Forecast - Tutti i Giorni")
with open("index.html", "w", encoding="utf-8") as f:
    f.write(final_html_main)

# --- GENERAZIONE PAGINE GIORNALIERE ---

for day in unique_days:
    # Filtra il DataFrame per il giorno corrente
    daily_df = df[df["time"].dt.date == day].copy()
    
    # Inizializza un NUOVO dizionario per ogni giorno
    combined_means_daily = {} 
    
    figs_daily = []
    for i in range(len(field_names)):
        figs_daily.append(get_series(daily_df, i, combined_means_daily))

    daily_combined_df = pd.DataFrame({"time": daily_df["time"]})
    for k, v in combined_means_daily.items():
        daily_combined_df[k] = v

    # Genera HTML
    day_name = day.strftime("%A %d-%m") 
    html_content_daily = f"<h1>Previsioni Dettagliate per {day_name}</h1>"
    
    combined_daily_fig = create_mid_graph(daily_df, daily_combined_df)
    html_content_daily += pio.to_html(combined_daily_fig, full_html=False, include_plotlyjs='cdn')
    
    html_content_daily += f"<h2 style='padding-top: 30px;'>Dettagli Modelli</h2>"
    for i in range(len(figs_daily)):
        html_content_daily += pio.to_html(figs_daily[i], full_html=False, include_plotlyjs=False)

    # SALVATAGGIO PAGINA GIORNALIERA (es. 2025-11-24.html)
    filename = f"{day.strftime('%Y-%m-%d')}.html"
    final_html_daily = create_final_html(html_content_daily, f"Meteo Forecast - {day_name}")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(final_html_daily)