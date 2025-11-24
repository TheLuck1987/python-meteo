import json
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import requests
import os
from datetime import datetime
from plotly.colors import qualitative

# --- CONFIGURAZIONE POSIZIONI ---
LOCATIONS = {
    "home": {"lat": 45.7256, "lon": 12.6897},
    "cs": {"lat": 45.7585, "lon": 12.8447},
    "rugby": {"lat": 45.6956, "lon": 12.7102}
}

# --- COSTANTI ---
source_names = ["_gfs_global","_ecmwf_ifs","_ecmwf_ifs025","_icon_global","_icon_d2",
                "_meteofrance_arpege_europe","_knmi_harmonie_arome_europe","_dmi_harmonie_arome_europe",
                "_italia_meteo_arpae_icon_2i","_bom_access_global","_cma_grapes_global"]
field_names = ["temperature_2m","precipitation","precipitation_probability","cape","wind_speed_10m",
               "wind_gusts_10m","surface_pressure","cloud_cover","relative_humidity_2m","dew_point_2m",
               "apparent_temperature","rain","showers","snowfall"]
serie_names_it = ["Temperatura","Precipitazioni","Prob. Prec.","CAPE","Vento","Raffiche","Pressione",
                  "Copertura Nuvolosa","Umidità Relativa","Punto di Rugiada","Temp. Percepita",
                  "Pioggia","Rovesci","Neve"]
DAY_NAMES_IT = {'Monday':'Lunedì','Tuesday':'Martedì','Wednesday':'Mercoledì','Thursday':'Giovedì',
                'Friday':'Venerdì','Saturday':'Sabato','Sunday':'Domenica'}
X_AXIS_TITLE = "Ora (Locale)"
TITLE_MAIN = "Previsioni Medie 7 Giorni"
TITLE_DETAIL = "Previsioni Dettagliate per Modello"
PAST_MEAN_LABEL = "MEDIA 50 ANNI"

# --- FUNZIONI ---
def get_data(latitude, longitude, local_path):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}"
           "&daily=wind_direction_10m_dominant&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,"
           "apparent_temperature,precipitation_probability,precipitation,cloud_cover,wind_speed_10m,"
           "wind_gusts_10m,wind_direction_10m,surface_pressure,cape,snowfall,showers,rain"
           "&models=ecmwf_ifs025,icon_d2,gfs_global,ecmwf_ifs,icon_global,gem_regional,"
           "knmi_harmonie_arome_europe,dmi_harmonie_arome_europe,meteofrance_arpege_europe,"
           "italia_meteo_arpae_icon_2i,bom_access_global,cma_grapes_global")
    r = requests.get(url)
    r.raise_for_status()
    with open(local_path,'wb') as f:
        f.write(r.content)

def robust_mean(arr):
    vals = np.array([v for v in arr if v is not None and not pd.isna(v)])
    if len(vals)==0: return np.nan
    q1,q3 = np.percentile(vals,[25,75])
    iqr = q3-q1
    filt = [v for v in vals if q1-1.5*iqr <= v <= q3+1.5*iqr]
    return np.mean(filt) if filt else np.median(vals)

def create_mid_graph(df, combined_data):
    fig = go.Figure()
    colors = qualitative.Light24
    for idx,col in enumerate(combined_data.columns[1:]):
        fig.add_trace(go.Scatter(x=df["time"],y=combined_data[col],mode="lines",
                                 name=col,line=dict(width=1.5,color=colors[idx%len(colors)])))
    tickvals = [t for t in df["time"] if t.hour==0]
    fig.update_xaxes(tickvals=tickvals,tickformat="%a %d-%m")
    fig.update_layout(template="plotly_dark",hovermode="x unified",height=650,xaxis_title=X_AXIS_TITLE)
    return fig

def get_series(df, serie_index, combined_means, pdf_indexed=None):
    valid_fields = [
        f"{field_names[serie_index]}{src}" 
        for src in source_names 
        if f"{field_names[serie_index]}{src}" in df.columns and df[f"{field_names[serie_index]}{src}"].notna().any()
    ]

    past_values = None
    if serie_index == 0 and pdf_indexed is not None:
        # chiave per mappare ora e giorno ignorando l'anno
        df['month_day_hour'] = df['time'].dt.strftime("%m-%d %H:%M")
        # map robust mean sui dati storici
        def get_historical_mean(key):
            if key not in pdf_indexed.index:
                return np.nan
            row = pdf_indexed.loc[key]
            all_vals = []
            for val in row:
                if isinstance(val, list):
                    all_vals.extend(val)
                elif not pd.isna(val):
                    all_vals.append(val)
            return robust_mean(all_vals) if all_vals else np.nan

        past_values = df['month_day_hour'].map(get_historical_mean)
        combined_means[PAST_MEAN_LABEL] = past_values

    if valid_fields:
        df["mean"] = df[valid_fields].apply(robust_mean, axis=1)
        combined_means[serie_names_it[serie_index]] = df["mean"]

    fig = go.Figure()
    if past_values is not None:
        fig.add_trace(go.Scatter(
            x=df["time"], y=past_values, mode="lines", name=PAST_MEAN_LABEL,
            line=dict(width=1.5, dash="longdash", color="red")
        ))

    for f in valid_fields:
        fig.add_trace(go.Scatter(
            x=df["time"], y=df[f], mode="lines", name=f.replace(field_names[serie_index], ""),
            line=dict(width=1.5)
        ))

    if valid_fields:
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["mean"], mode="lines", name=serie_names_it[serie_index],
            line=dict(width=3, color="blue")
        ))

    return fig

def create_final_html(content,title,timestamp_html):
    return f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<title>{title}</title>
</head>
<body>
{content}
{timestamp_html}
</body>
</html>"""

def generate_forecast_pages(folder_name, latitude, longitude):
    os.makedirs(folder_name,exist_ok=True)
    json_path = os.path.join(folder_name,"open-meteo.json")
    get_data(latitude,longitude,json_path)
    with open(json_path,"r",encoding="utf_8") as f:
        data=json.load(f)
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"],utc=True).dt.tz_convert("Europe/Rome")
    unique_days = df["time"].dt.date.unique()

    # --- storico ottimizzato ---
    with open("historical.json","r",encoding="utf_8") as pf:
        pdf = pd.DataFrame(json.load(pf)["hourly"])
    pdf["time"] = pd.to_datetime(pdf["time"],utc=True).dt.tz_convert("Europe/Rome")
    pdf["year"] = pdf["time"].dt.year
    pdf["month_day_hour"] = pdf["time"].dt.strftime("%m-%d %H:%M")
    pdf_indexed = pdf.pivot_table(index="month_day_hour", columns="year",
                                  values="temperature_2m", aggfunc=list)

    timestamp_html = f'<div style="position:fixed;bottom:5px;right:10px;font-size:0.7em;color:#999;">Ultimo aggiornamento: {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}</div>'

    # --- main page ---
    combined_means_main = {}
    figs_combined = [get_series(df,i,combined_means_main,pdf_indexed if i==0 else None)
                     for i in range(len(field_names))]
    combined_df_main = pd.DataFrame({"time":df["time"]})
    for k,v in combined_means_main.items():
        combined_df_main[k] = v
    html_content_main = f"<h1>{TITLE_MAIN} - {folder_name.upper()}</h1>"
    html_content_main += pio.to_html(create_mid_graph(df,combined_df_main),full_html=False,include_plotlyjs='cdn')
    html_content_main += f"<h1>{TITLE_DETAIL}</h1>"
    for fig in figs_combined:
        html_content_main += pio.to_html(fig,full_html=False,include_plotlyjs=False)
    with open(os.path.join(folder_name,"index.html"),"w",encoding="utf-8") as f:
        f.write(create_final_html(html_content_main,f"Meteo Forecast - {folder_name.upper()}",timestamp_html))

    # --- pagine giornaliere ---
    for idx, day in enumerate(unique_days):
        daily_df = df[df["time"].dt.date==day].copy()
        combined_means_daily = {}
        figs_daily = [get_series(daily_df,i,combined_means_daily,pdf_indexed if i==0 else None)
                      for i in range(len(field_names))]
        daily_combined_df = pd.DataFrame({"time":daily_df["time"]})
        for k,v in combined_means_daily.items():
            daily_combined_df[k] = v
        day_name_it = DAY_NAMES_IT.get(day.strftime("%A"), day.strftime("%A"))
        html_content_daily = f"<h1>Previsioni Dettagliate per {day_name_it} {day.strftime('%d-%m')}</h1>"
        html_content_daily += pio.to_html(create_mid_graph(daily_df,daily_combined_df),full_html=False,include_plotlyjs='cdn')
        for fig in figs_daily:
            html_content_daily += pio.to_html(fig,full_html=False,include_plotlyjs=False)
        with open(os.path.join(folder_name,f"{idx}.html"),"w",encoding="utf-8") as f:
            f.write(create_final_html(html_content_daily,
                                      f"Meteo Forecast - {folder_name.upper()} - {day_name_it}",
                                      timestamp_html))

# --- ESECUZIONE ---
if __name__=="__main__":
    for folder,cfg in LOCATIONS.items():
        generate_forecast_pages(folder,cfg["lat"],cfg["lon"])
