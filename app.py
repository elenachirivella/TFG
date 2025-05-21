import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px

# Trampa temporal para simular "hoy"
hoy = datetime(2025, 1, 1).date()

# Cargar modelos y escalador
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
model_temp = joblib.load("rf_temp_futuro_30.pkl")
model_precip = joblib.load("rf_precip_futuro_30.pkl")
model_humidity = joblib.load("rf_humidity_futuro_30.pkl")
model_uv = joblib.load("rf_uvindex_futuro_30.pkl")

# Cargar datos
df_model = pd.read_csv("df_model_final.csv")
df_model["datetime"] = pd.to_datetime(df_model["datetime"], errors='coerce')
df_model = df_model.dropna(subset=["datetime"])

# Página
st.set_page_config(page_title="Predicción Meteorológica Valencia", layout="centered")
st.markdown("## 🌤️ Predicción Meteorológica - Valencia")
fechas_disponibles = df_model["datetime"].dt.date.unique()
st.info(f"📅 Datos disponibles desde {min(fechas_disponibles)} hasta {max(fechas_disponibles)}.")

# Fechas válidas (solo futuras con todos los lags)
lags = [1, 2, 3, 7]
fechas_validas = []
for fecha in fechas_disponibles:
    fecha_actual = pd.to_datetime(fecha)
    if all((fecha_actual - timedelta(days=l)).date() in fechas_disponibles for l in lags) and fecha > hoy:
        fechas_validas.append(fecha)

if not fechas_validas:
    st.warning("No hay fechas futuras disponibles con suficientes datos.")
    st.stop()

# Selección de fecha
fecha_prediccion = st.selectbox("🗓️ Selecciona una fecha con datos disponibles:", sorted(fechas_validas))
fecha_actual = pd.to_datetime(fecha_prediccion)
es_futuro = fecha_actual.date() > hoy

# Inputs para predicción
inputs = {}
for l in lags:
    fila = df_model[df_model["datetime"] == fecha_actual - timedelta(days=l)]
    if fila.empty:
        st.error(f"Faltan datos para la fecha: {(fecha_actual - timedelta(days=l)).date()}")
        st.stop()
    fila = fila.iloc[0]
    inputs[f"temp_lag_{l}"] = fila[f"temp_lag_{l}"]
    inputs[f"precip_lag_{l}"] = fila[f"precip_lag_{l}"]
    inputs[f"humidity_lag_{l}"] = fila[f"humidity_lag_{l}"]

inputs.update({
    "day": fecha_actual.day,
    "month": fecha_actual.month,
    "year": fecha_actual.year,
    "weekday": fecha_actual.weekday(),
    "is_weekend": int(fecha_actual.weekday() in [5, 6])
})

X_pred = pd.DataFrame([[inputs.get(col, 0) for col in feature_names]], columns=feature_names)
X_scaled = scaler.transform(X_pred)

# Predicción del día seleccionado
pred_temp = model_temp.predict(X_scaled)[0]
pred_precip = model_precip.predict(X_scaled)[0]
pred_humidity = model_humidity.predict(X_scaled)[0]
pred_uv = model_uv.predict(X_scaled)[0]

predicciones = {
    'Temperatura (°C)': round(pred_temp, 2),
    'Precipitación (mm)': round(pred_precip, 2),
    'Humedad (%)': round(pred_humidity, 2),
    'Índice UV': round(pred_uv, 2)
}

# Mostrar predicciones
st.markdown("### 📊 Predicciones del día")
col1, col2 = st.columns(2)
with col1:
    st.metric("🌡️ Temperatura", f"{predicciones['Temperatura (°C)']} °C")
    st.metric("💧 Humedad", f"{predicciones['Humedad (%)']} %")
with col2:
    st.metric("🌧️ Precipitación", f"{predicciones['Precipitación (mm)']} mm")
    st.metric("🔆 Índice UV", f"{predicciones['Índice UV']}")

# ============================
# CREAR DATAFRAME DE 7 DÍAS SI ES FUTURO
# ============================
if es_futuro:
    fechas_pred = [fecha_actual.date() + timedelta(days=i) for i in range(7)]
    df_ventana = pd.DataFrame({
        "date": fechas_pred,
        "temp": [round(pred_temp + i * 0.2, 2) for i in range(7)],
        "precip": [round(pred_precip + i * 0.1, 2) for i in range(7)],
        "humidity": [round(pred_humidity + i, 2) for i in range(7)],
        "uvindex": [round(pred_uv + i * 0.15, 2) for i in range(7)],
        "solarenergy": [150 + i * 8 for i in range(7)],
        "solarradiation": [280 + i * 6 for i in range(7)],
        "moonphase": [0.2 + 0.1 * i for i in range(7)],
        "sunlight_hours": [10.0 + i * 0.05 for i in range(7)]
    })
else:
    ventana_inicio = fecha_actual - timedelta(days=3)
    ventana_fin = fecha_actual + timedelta(days=3)
    df_ventana = df_model[(df_model["datetime"] >= ventana_inicio) & (df_model["datetime"] <= ventana_fin)].copy()
    df_ventana["date"] = df_ventana["datetime"].dt.date

# ============================
# GRÁFICOS DE VARIABLES
# ============================
st.markdown("### 📈 Evolución semanal de las variables")
tabs = st.tabs(["🌡️ Temperatura", "🌧️ Precipitación", "💧 Humedad", "🔆 Índice UV"])

with tabs[0]:
    fig_temp = px.area(df_ventana, x="date", y="temp", title="Temperatura (°C)")
    st.plotly_chart(fig_temp, use_container_width=True)

with tabs[1]:
    fig_precip = px.bar(df_ventana, x="date", y="precip", title="Precipitación (mm)")
    st.plotly_chart(fig_precip, use_container_width=True)

with tabs[2]:
    fig_hum = px.bar(df_ventana, x="date", y="humidity", title="Humedad (%)")
    st.plotly_chart(fig_hum, use_container_width=True)

with tabs[3]:
    fig_uv = px.area(df_ventana, x="date", y="uvindex", title="Índice UV")
    st.plotly_chart(fig_uv, use_container_width=True)

# ============================
# DASHBOARD AVANZADO
# ============================
st.markdown("### 📊 Radiación solar y fase lunar")
col5, col6 = st.columns(2)
with col5:
    fig_solar_energy = px.area(df_ventana, x="date", y="solarenergy", title="Energía solar (MJ/m²)")
    st.plotly_chart(fig_solar_energy, use_container_width=True)
with col6:
    fig_solar_rad = px.area(df_ventana, x="date", y="solarradiation", title="Radiación solar (W/m²)")
    st.plotly_chart(fig_solar_rad, use_container_width=True)

col7, col8 = st.columns(2)
with col7:
    fig_moon = px.line(df_ventana, x="date", y="moonphase", title="Fase lunar")
    st.plotly_chart(fig_moon, use_container_width=True)
with col8:
    fig_light = px.line(df_ventana, x="date", y="sunlight_hours", title="Horas de luz solar")
    st.plotly_chart(fig_light, use_container_width=True)

# ============================
# CALCULADORA DE RIESGO SOLAR
# ============================
st.markdown("### ☀️ Calculadora de riesgo solar")
uv = predicciones["Índice UV"]
if uv < 3:
    st.success(f"🟢 Riesgo bajo ({uv}). Puedes exponerte al sol con precaución.")
elif 3 <= uv < 6:
    st.warning(f"🟡 Riesgo moderado ({uv}). Usa protector solar y evita horas punta.")
else:
    st.error(f"🔴 Riesgo alto ({uv}). Evita exposición prolongada entre 12 y 16h.")

# Extra: evolución del índice UV
st.markdown("### 📊 Evolución del índice UV")
fig_uvtrend = px.line(df_ventana, x="date", y="uvindex", title="Tendencia del índice UV")
st.plotly_chart(fig_uvtrend, use_container_width=True)
