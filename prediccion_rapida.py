# =============================================================================
# PREDICCIÓN RÁPIDA — MERCADO ELÉCTRICO ESPAÑOL
# =============================================================================
# Usa el modelo ya entrenado (pkl) para predecir el día siguiente.
# No necesita reentrenar — carga directamente el modelo guardado.
#
# Requisito: haber ejecutado mercado_electrico.py al menos una vez para
# generar los archivos:
#   - modelo_mercado_electrico.pkl
#   - features_mercado_electrico.pkl
# =============================================================================

import requests
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
MODELO_PATH   = "modelo_mercado_electrico.pkl"
FEATURES_PATH = "features_mercado_electrico.pkl"

CIUDADES = {
    'Galicia':   {'lat': 43.37, 'lon': -8.39},
    'Zaragoza':  {'lat': 41.65, 'lon': -0.88},
    'Andalucia': {'lat': 37.38, 'lon': -5.98},
}

# ── 1. CARGAR MODELO ──────────────────────────────────────────────────────────
print("📦 Cargando modelo guardado...")
modelo       = joblib.load(MODELO_PATH)
feature_cols = joblib.load(FEATURES_PATH)
print("  ✅ Modelo listo")

# ── 2. PRECIO DE AYER Y SEMANA PASADA (para los lags) ────────────────────────
def obtener_precios_recientes(horas: int = 200) -> pd.Series:
    """Descarga las últimas N horas de precios de REE para calcular lags."""
    url = "https://apidatos.ree.es/es/datos/mercados/precios-mercados-tiempo-real"
    ahora  = datetime.now()
    inicio = ahora - timedelta(hours=horas)
    params = {
        'start_date': inicio.strftime('%Y-%m-%dT%H:%M'),
        'end_date':   ahora.strftime('%Y-%m-%dT%H:%M'),
        'time_trunc': 'hour'
    }
    r = requests.get(url, params=params,
                     headers={'Accept': 'application/json'}, timeout=15)
    datos = [{'Fecha': v['datetime'], 'Precio': v['value']}
             for v in r.json()['included'][0]['attributes']['values']]
    df = pd.DataFrame(datos)
    df['Fecha'] = pd.to_datetime(df['Fecha'], utc=True)
    df = df.set_index('Fecha')
    df.index = df.index.tz_convert('Europe/Madrid').tz_localize(None)
    return df['Precio'].sort_index()

print("⚡ Descargando precios recientes para lags...")
precios_recientes = obtener_precios_recientes(200)
print(f"  ✅ {len(precios_recientes)} horas recientes obtenidas")

# ── 3. FORECAST METEOROLÓGICO ─────────────────────────────────────────────────
manana    = datetime.now() + timedelta(days=1)
fecha_str = manana.strftime('%Y-%m-%d')
print(f"\n🌤️  Descargando previsión meteorológica para {fecha_str}...")

lats = ",".join(str(c['lat']) for c in CIUDADES.values())
lons = ",".join(str(c['lon']) for c in CIUDADES.values())

r = requests.get("https://api.open-meteo.com/v1/forecast", params={
    "latitude": lats, "longitude": lons,
    "start_date": fecha_str, "end_date": fecha_str,
    "hourly": ["temperature_2m", "shortwave_radiation", "wind_speed_10m"],
    "timezone": "Europe/Madrid"
}, timeout=30).json()

horas  = pd.to_datetime(r[0]['hourly']['time'])
df_fc  = pd.DataFrame(index=horas)

for i, ciudad in enumerate(CIUDADES.keys()):
    df_fc[f'Temp_{ciudad}']      = r[i]['hourly']['temperature_2m']
    df_fc[f'Radiacion_{ciudad}'] = r[i]['hourly']['shortwave_radiation']
    df_fc[f'Viento_{ciudad}']    = r[i]['hourly']['wind_speed_10m']

print("  ✅ Previsión meteorológica lista")

# ── 4. DEMANDA PREVISTA ───────────────────────────────────────────────────────
try:
    params_dem = {
        'start_date': manana.strftime('%Y-%m-%dT00:00'),
        'end_date':   manana.strftime('%Y-%m-%dT23:59'),
        'time_trunc': 'hour'
    }
    r_dem = requests.get(
        "https://apidatos.ree.es/es/datos/demanda/demanda-tiempo-real",
        params=params_dem, headers={'Accept': 'application/json'}, timeout=15
    ).json()
    demanda_vals = []
    for curva in r_dem.get('included', []):
        if any(k in curva.get('type', '').lower() for k in ['prevista', 'programada', 'real']):
            demanda_vals = [v['value'] for v in curva['attributes']['values']]
            break
    df_fc['Demanda'] = demanda_vals[:24] if demanda_vals else precios_recientes.mean()
except:
    # Fallback: media histórica por hora
    df_fc['Demanda'] = 25000

# ── 5. VARIABLES DE CALENDARIO Y LAGS ────────────────────────────────────────
df_fc['Hora']            = df_fc.index.hour
df_fc['Dia_Semana']      = manana.weekday()
df_fc['Mes']             = manana.month
df_fc['Es_FinDeSemana']  = int(manana.weekday() >= 5)

# Lags de precio usando historial reciente
df_fc['Precio_Lag_1h']   = precios_recientes.iloc[-1]
df_fc['Precio_MA_24h']   = precios_recientes.iloc[-24:].mean()

if len(precios_recientes) >= 24:
    df_fc['Precio_Lag_24h'] = precios_recientes.iloc[-24:].values
else:
    df_fc['Precio_Lag_24h'] = precios_recientes.mean()

if len(precios_recientes) >= 168:
    df_fc['Precio_Lag_168h'] = precios_recientes.iloc[-168:-144].values
else:
    df_fc['Precio_Lag_168h'] = precios_recientes.mean()

# ── 6. PREDICCIÓN ─────────────────────────────────────────────────────────────
df_fc = df_fc[feature_cols]
precios_pred          = modelo.predict(df_fc)
df_fc['Precio_Pred']  = precios_pred

# ── 7. PANEL INDUSTRIAL ───────────────────────────────────────────────────────
precios       = df_fc['Precio_Pred']
precio_medio  = precios.mean()
horas_baratas = precios.nsmallest(6).index
horas_caras   = precios.nlargest(6).index
diferencial   = precios.nlargest(6).mean() - precios.nsmallest(6).mean()

print("\n" + "="*60)
print(f"PANEL INDUSTRIAL — {manana.strftime('%d/%m/%Y')}")
print("="*60)
print(f"  📊 Precio medio estimado: {precio_medio:.1f} €/MWh")
print(f"\n  ✅ ENCENDER MÁQUINAS (horas más baratas):")
for h in sorted(horas_baratas):
    print(f"     {h.strftime('%H:%M')} → {precios[h]:.1f} €/MWh")
print(f"\n  ❌ APAGAR MÁQUINAS (horas más caras):")
for h in sorted(horas_caras):
    print(f"     {h.strftime('%H:%M')} → {precios[h]:.1f} €/MWh")
print(f"\n  💰 Diferencial pico/valle: {diferencial:.1f} €/MWh")

# ── 8. GRÁFICA ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
colores = ['#E74C3C' if h in horas_caras
           else '#2ECC71' if h in horas_baratas
           else '#BDC3C7' for h in precios.index]

ax.bar(precios.index.hour, precios.values, color=colores,
       edgecolor='white', linewidth=0.5)
ax.axhline(precio_medio, color='#2C3E50', linewidth=1.5, linestyle='--',
           label=f'Precio medio: {precio_medio:.1f} €/MWh')
ax.set_title(f"Panel de Control Industrial — {manana.strftime('%d/%m/%Y')}",
             fontsize=14, fontweight='bold')
ax.set_xlabel('Hora del día', fontsize=12)
ax.set_ylabel('Precio estimado €/MWh', fontsize=12)
ax.set_xticks(range(24))

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color='#2ECC71', label='Encender máquinas (más barato)'),
    Patch(color='#E74C3C', label='Apagar máquinas (más caro)'),
    Patch(color='#BDC3C7', label='Horas intermedias'),
] + [ax.get_legend_handles_labels()[0][0]], loc='upper left')

ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('panel_industrial_manana.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  💾 Panel guardado en: panel_industrial_manana.png")
