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

# --- CONFIGURAZIONE POSIZIONI ---

LOCATIONS = {
    "home": {"lat": 45.7256, "lon": 12.6897},
    "cs": {"lat": 45.7585, "lon": 12.8447},
    "rugby": {"lat": 45.6956, "lon": 12.7102}
}


# --- COSTANTI E TRADUZIONI ---

source_names = [ "_gfs_global", "_ecmwf_ifs", "_ecmwf_ifs025", "_icon_global", "_icon_d2", "_meteofrance_arpege_europe", "_knmi_harmonie_arome_europe", "_dmi_harmonie_arome_europe", "_italia_meteo_arpae_icon_2i", "_bom_access_global", "_cma_grapes_global" ]
friendly_sources = [ "GFS", "ECMWF", "ECMWF_025", "ICON", "ICON_D2", "ARPEGE", "KNMI", "DMI", "ARPAE", "BOM", "CMA" ]
field_names = [ "temperature_2m", "precipitation", "precipitation_probability", "cape", "wind_speed_10m", "wind_gusts_10m", "surface_pressure", "cloud_cover", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "rain", "showers", "snowfall" ]

# TRADUZIONI IN ITALIANO
serie_names_it = [ "Temperatura", "Precipitazioni", "Prob. Prec.", "CAPE", "Vento", "Raffiche", "Pressione", "Copertura Nuvolosa", "Umidità Relativa", "Punto di Rugiada", "Temp. Percepita", "Pioggia", "Rovesci", "Neve" ]
titles_it = ["Temperatura [°C]", "Precipitazioni [mm/h]", "Probabilità di Precipitazioni [%]", "Indice CAPE [J/kg]", "Velocità del Vento [km/h]", "Raffiche di Vento [km/h]", "Pressione [hPa]", "Copertura Nuvolosa [%]", "Umidità Relativa [%]", "Punto di Rugiada [°C]", "Temperatura Percepita [°C]", "Pioggia [mm/h]", "Rovesci [mm/h]", "Neve [cm/h]" ]

# TRADUZIONE NOMI GIORNI
DAY_NAMES_IT = {
    'Monday': 'Lunedì', 'Tuesday': 'Martedì', 'Wednesday': 'Mercoledì', 
    'Thursday': 'Giovedì', 'Friday': 'Venerdì', 'Saturday': 'Sabato', 
    'Sunday': 'Domenica'
}

# TITOLI E ASSI GENERALI
X_AXIS_TITLE = "Ora (Locale)"
TITLE_MAIN = "Previsioni Medie 7 Giorni"
TITLE_DETAIL = "Previsioni Dettagliate per Modello"
PAST_MEAN_LABEL = "MEDIA 50 ANNI"


# --- FUNZIONI DI BASE ---

def get_data(latitude, longitude, local_path):
    """Scarica i dati meteo per le coordinate e salva nel percorso specificato."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&daily=wind_direction_10m_dominant&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,precipitation,cloud_cover,wind_speed_10m,wind_gusts_10m,wind_direction_10m,surface_pressure,cape,snowfall,showers,rain&models=ecmwf_ifs025,icon_d2,gfs_global,ecmwf_ifs,icon_global,gem_regional,knmi_harmonie_arome_europe,dmi_harmonie_arome_europe,meteofrance_arpege_europe,italia_meteo_arpae_icon_2i,bom_access_global,cma_grapes_global"
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
    """Calcola la media storica robusta per un dato punto temporale."""
    values = []
    for y in range(1974, 2025):
        var_date = date.replace(year = y)
        match = pdf.loc[pdf["time"] == var_date, "temperature_2m"]
        if not match.empty:
            values.append(match.iloc[0])
    return robust_mean(values)

# --- FUNZIONI DI GENERAZIONE GRAFICI E HTML ---

def create_mid_graph(df, combined_data): 
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

def get_series(df, serie_index, current_combined_means, pdf_past, past_values_cache): 
    """Crea il grafico di una singola serie. Accetta il PDF storico e la cache."""
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
        # TENTA IL RECUPERO DALLA CACHE
        cache_key = tuple(df["time"].dt.strftime('%Y-%m-%d %H:%M').tolist())
        
        if cache_key in past_values_cache:
            # Recupero immediato dalla cache (Veloce!)
            past_values = past_values_cache[cache_key]
        else:
            # Calcolo pesante solo se non in cache
            past_values = []
            for d in df["time"]:
                past_values.append(get_past(d, pdf_past)) # Usa PDF pre-caricato
            # Salva il risultato nella cache
            past_values_cache[cache_key] = past_values
            
        current_combined_means[PAST_MEAN_LABEL] = past_values.copy() 
        
    if valid_fields:
        df["mean"] = df[valid_fields].apply(robust_mean, axis=1)
        current_combined_means[serie_names_it[serie_index]] = df["mean"].copy() 
        
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
            mode="lines", name=serie_names_it[serie_index], 
            line=dict(width=3, dash="solid", color="blue"),
            visible=True 
        ))   
        
    fig.update_layout(
        title=titles_it[serie_index], 
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

def create_final_html(content, title, timestamp_html):
    """Genera la struttura HTML completa con header, CSS, script di refresh e timestamp."""
    return f"""
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script>
        function refreshPage() {{
            console.log('Eseguo il refresh della pagina...');
            window.location.reload(true);
            setPageRefresh();
        }}
        function setPageRefresh() {{
            let now = new Date();
            let currentSeconds = now.getMinutes() * 60 + now.getSeconds() + now.getMilliseconds() / 1000;
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
        .day-link {{ color: #4CAF50; margin-right: 20px; font-size: 1.1em; text-decoration: none; }}
        .day-link:hover {{ text-decoration: underline; }}
        /* Stile per il timestamp */
        .timestamp {{ position: fixed; bottom: 5px; right: 10px; font-size: 0.7em; color: #999; z-index: 1000; }}
    </style>
</head>
<body>
{content}
{timestamp_html} 
<script>setPageRefresh()</script>
</body>
</html>
"""

# --- FUNZIONE PRINCIPALE DI ELABORAZIONE ---

def generate_forecast_pages(folder_name, latitude, longitude, pdf_past, past_values_cache):
    """Esegue l'intero processo di generazione per una singola posizione."""
    
    print(f"--- Elaborazione iniziata per la cartella: {folder_name} ---")

    # 1. Creazione della cartella e download dati
    os.makedirs(folder_name, exist_ok=True)
    json_path = os.path.join(folder_name, "open-meteo.json")
    get_data(latitude, longitude, json_path)
    
    # Genera il timestamp di elaborazione
    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    timestamp_html = f'<div class="timestamp">Ultimo aggiornamento: {now}</div>'

    # 2. Caricamento e preparazione dati
    with open(json_path, "r", encoding="utf_8") as f:
        data = json.load(f)  
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Rome")
    df = df[df['time'] >= df.iloc[0]['time'].normalize() + pd.Timedelta(hours=datetime.now().hour)]
    unique_days = df["time"].dt.date.unique()

    # --- GENERAZIONE PAGINA PRINCIPALE (index.html: TUTTI I GIORNI) ---

    combined_means_main = {}
    figs_combined = []

    # PASSA LA CACHE E PDF_PAST
    for i in range(len(field_names)):
        figs_combined.append(get_series(df, i, combined_means_main, pdf_past, past_values_cache)) 

    combined_df_main = pd.DataFrame({"time": df["time"]})
    for k, v in combined_means_main.items():
        combined_df_main[k] = v

    # --- CREAZIONE LINK DI NAVIGAZIONE ---
    link_html = "<h1 style='padding-top: 30px;'>Navigazione Giornaliera</h1>"
    link_html += "<div style='padding: 10px;'>"
    for idx, day in enumerate(unique_days):
        day_name_en = day.strftime("%A")
        day_name_it = DAY_NAMES_IT.get(day_name_en, day_name_en)
        filename = f"{idx}.html"
        link_label = f"{day_name_it} {day.strftime('%d-%m')}"
        link_html += f"<a href='{filename}' class='day-link'>{link_label}</a>"
    link_html += "</div>"
    # --- FINE CREAZIONE LINK ---


    html_content_main = f"<h1>{TITLE_MAIN} - {folder_name.upper()}</h1>"
    html_content_main += link_html 

    combined_fig = create_mid_graph(df, combined_df_main) 
    html_content_main += pio.to_html(combined_fig, full_html=False, include_plotlyjs='cdn')
    html_content_main += f"<h1 style='padding-top: 30px;'>{TITLE_DETAIL}</h1>"
    for i in range(len(figs_combined)):
        html_content_main += pio.to_html(figs_combined[i], full_html=False, include_plotlyjs=False)

    # SALVATAGGIO PAGINA PRINCIPALE
    final_html_main = create_final_html(html_content_main, f"Meteo Forecast - {folder_name.upper()} - Tutti i Giorni", timestamp_html)
    with open(os.path.join(folder_name, "index.html"), "w", encoding="utf-8") as f:
        f.write(final_html_main)

    # --- GENERAZIONE PAGINE GIORNALIERE (0.html, 1.html, ...) ---

    for idx, day in enumerate(unique_days):
        daily_df = df[df["time"].dt.date == day].copy()
        
        combined_means_daily = {} 
        figs_daily = []
        
        for i in range(len(field_names)):
            # PASSA LA CACHE E PDF_PAST
            figs_daily.append(get_series(daily_df, i, combined_means_daily, pdf_past, past_values_cache))

        daily_combined_df = pd.DataFrame({"time": daily_df["time"]})
        for k, v in combined_means_daily.items():
            daily_combined_df[k] = v

        # Nome del giorno tradotto per il titolo
        day_name_en = day.strftime("%A")
        day_name_it = DAY_NAMES_IT.get(day_name_en, day_name_en)
        
        day_label_it = f"{day_name_it} {day.strftime('%d-%m')}"
        
        # Genera HTML
        html_content_daily = f"<h1>Previsioni Dettagliate per {day_label_it} ({folder_name.upper()})</h1>"
        
        # Aggiunge un link per tornare all'indice principale
        html_content_daily += f"<p style='padding: 10px;'><a href='index.html' class='day-link'>← Torna alla panoramica 7 giorni</a></p>"

        combined_daily_fig = create_mid_graph(daily_df, daily_combined_df) 
        html_content_daily += pio.to_html(combined_daily_fig, full_html=False, include_plotlyjs='cdn')
        
        html_content_daily += f"<h2 style='padding-top: 30px;'>Dettagli Modelli</h2>"
        for i in range(len(figs_daily)):
            html_content_daily += pio.to_html(figs_daily[i], full_html=False, include_plotlyjs=False)

        # SALVATAGGIO PAGINA GIORNALIERA (0.html, 1.html, ...)
        filename = f"{idx}.html"
        final_html_daily = create_final_html(html_content_daily, f"Meteo Forecast - {folder_name.upper()} - {day_label_it}", timestamp_html)
        with open(os.path.join(folder_name, filename), "w", encoding="utf-8") as f:
            f.write(final_html_daily)
            
    print(f"--- Elaborazione completata per la cartella: {folder_name} ---\n")


# --- ESECUZIONE GLOBALE (con pre-caricamento e caching) ---

if __name__ == "__main__":
    
    # 1. PRE-CARICAMENTO DATI STORICI (Eseguito UNA SOLA VOLTA)
    print("Pre-caricamento dati storici (historical.json)...")
    with open("historical.json", "r", encoding="utf_8") as pf:
        past_data = json.load(pf)
    PDF_PAST = pd.DataFrame(past_data["hourly"])
    PDF_PAST["time"] = pd.to_datetime(PDF_PAST["time"], utc=True).dt.tz_convert("Europe/Rome")
    print("Pre-caricamento completato.")
    
    # Dizionario per la cache dei risultati (Eseguito UNA SOLA VOLTA)
    PAST_VALUES_CACHE = {}

    # 2. ESECUZIONE PER OGNI POSIZIONE
    for folder, config in LOCATIONS.items():
        # Passiamo lo stesso PDF pre-caricato e la stessa cache a tutte le chiamate
        generate_forecast_pages(folder, config["lat"], config["lon"], PDF_PAST, PAST_VALUES_CACHE)