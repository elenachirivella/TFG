import streamlit as st
import pandas as pd
import joblib
from datetime import date, timedelta, datetime
import numpy as np
import plotly.express as px

# ⚠️ Trampa temporal para simular que hoy es una fecha del dataset
hoy = datetime(2025, 1, 1).date()

# Cargar scaler y columnas usadas en el entrenamiento
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Cargar modelos
model_temp = joblib.load("rf_temp_futuro_30.pkl")
model_precip = joblib.load("rf_precip_futuro_30.pkl")
model_humidity = joblib.load("rf_humidity_futuro_30.pkl")
model_uv = joblib.load("rf_uvindex_futuro_30.pkl")

# Cargar dataset
df_model = pd.read_csv("df_model_final.csv")
df_model["datetime"] = pd.to_datetime(df_model["datetime"], errors='coerce')
df_model = df_model.dropna(subset=["datetime"])

# Cabecera visual
st.set_page_config(page_title="Predicción Meteorológica Valencia", layout="centered")
st.markdown("## 🌤️ Predicción Meteorológica - Valencia")
st.markdown("Esta aplicación predice el clima de Valencia para cualquier fecha con datos disponibles.")

# === Buscar fechas válidas para predicción ===
fechas_validas = []
lags = [1, 2, 3, 7]
fechas_disponibles = df_model["datetime"].dt.date.unique()

for fecha in fechas_disponibles:
    fecha_actual = pd.to_datetime(fecha)
    faltan_lags = False
    for l in lags:
        if (fecha_actual - timedelta(days=l)).date() not in fechas_disponibles:
            faltan_lags = True
            break
    if not faltan_lags:
        fechas_validas.append(fecha)

# Mostrar rango total de fechas disponibles
st.info(f"📅 Datos disponibles desde {min(fechas_disponibles)} hasta {max(fechas_disponibles)}.")

# Clasificar fechas
fechas_pasadas = [f for f in fechas_validas if f <= hoy]
fechas_futuras = [f for f in fechas_validas if f > hoy]

# Permitir elegir entre fechas futuras o pasadas
tipo_fecha = st.radio("¿Qué fechas quieres consultar?", ["Futuras", "Pasadas"])

# Mostrar selector correspondiente
if tipo_fecha == "Futuras":
    opciones = sorted(fechas_futuras)
else:
    opciones = sorted(fechas_pasadas)

if not opciones:
    st.warning(f"No hay fechas {'futuras' if tipo_fecha == 'Futuras' else 'pasadas'} disponibles con suficientes datos.")
    st.stop()

fecha_prediccion = st.selectbox("🗓️ Selecciona una fecha con datos disponibles:", opciones)

# Preparar datos de entrada
fecha_actual = pd.to_datetime(fecha_prediccion)
inputs = {}

for l in lags:
    fecha_lag = fecha_actual - timedelta(days=l)
    fila = df_model[df_model["datetime"] == fecha_lag]
    if fila.empty:
        st.error(f"Faltan datos para la fecha: {fecha_lag.date()}. No se puede predecir.")
        st.stop()
    inputs[f"temp_lag_{l}"] = fila.iloc[0][f"temp_lag_{l}"]
    inputs[f"precip_lag_{l}"] = fila.iloc[0][f"precip_lag_{l}"]
    inputs[f"humidity_lag_{l}"] = fila.iloc[0][f"humidity_lag_{l}"]

inputs["day"] = fecha_actual.day
inputs["month"] = fecha_actual.month
inputs["year"] = fecha_actual.year
inputs["weekday"] = fecha_actual.weekday()
inputs["is_weekend"] = int(inputs["weekday"] in [5, 6])

# Crear DataFrame en el orden esperado
X_pred = pd.DataFrame([[inputs.get(col, 0) for col in feature_names]], columns=feature_names)
X_scaled = scaler.transform(X_pred)

# Predicciones
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

# Mostrar métricas
st.markdown("### 📊 Predicciones del día")
col1, col2 = st.columns(2)
with col1:
    st.metric("🌡️ Temperatura", f"{predicciones['Temperatura (°C)']} °C")
    st.metric("💧 Humedad", f"{predicciones['Humedad (%)']} %")
with col2:
    st.metric("🌧️ Precipitación", f"{predicciones['Precipitación (mm)']} mm")
    st.metric("🔆 Índice UV", f"{predicciones['Índice UV']}")

# Subconjunto ±3 días
ventana_inicio = fecha_actual - timedelta(days=3)
ventana_fin = fecha_actual + timedelta(days=3)
df_ventana = df_model[(df_model["datetime"] >= ventana_inicio) & (df_model["datetime"] <= ventana_fin)].copy()
df_ventana["date"] = df_ventana["datetime"].dt.date

# Visualización por variable
st.markdown("### 📈 Evolución semanal de las variables")
tabs = st.tabs(["🌡️ Temperatura", "🌧️ Precipitación", "💧 Humedad", "🔆 Índice UV"])

with tabs[0]:
    fig_temp = px.line(df_ventana, x="date", y="temp", title="Temperatura (°C)")
    st.plotly_chart(fig_temp, use_container_width=True)

with tabs[1]:
    fig_precip = px.line(df_ventana, x="date", y="precip", title="Precipitación (mm)")
    st.plotly_chart(fig_precip, use_container_width=True)

with tabs[2]:
    fig_hum = px.line(df_ventana, x="date", y="humidity", title="Humedad (%)")
    st.plotly_chart(fig_hum, use_container_width=True)

with tabs[3]:
    fig_uv = px.line(df_ventana, x="date", y="uvindex", title="Índice UV")
    st.plotly_chart(fig_uv, use_container_width=True)

# Dashboard avanzado
with st.expander("📊 Dashboard avanzado"):
    st.subheader("🔆 Radiación y Energía Solar")
    col5, col6 = st.columns(2)
    with col5:
        fig_solar_energy = px.line(df_ventana, x="date", y="solarenergy", title="Energía solar (MJ/m²)")
        st.plotly_chart(fig_solar_energy, use_container_width=True)
    with col6:
        fig_solar_rad = px.line(df_ventana, x="date", y="solarradiation", title="Radiación solar (W/m²)")
        st.plotly_chart(fig_solar_rad, use_container_width=True)

    st.subheader("🌙 Fase lunar y duración del día")
    col7, col8 = st.columns(2)
    with col7:
        fig_moon = px.line(df_ventana, x="date", y="moonphase", title="Fase lunar")
        st.plotly_chart(fig_moon, use_container_width=True)
    with col8:
        df_ventana["sunrise"] = pd.to_datetime(df_ventana["sunrise"], errors='coerce')
        df_ventana["sunset"] = pd.to_datetime(df_ventana["sunset"], errors='coerce')
        df_ventana["sunlight_hours"] = (df_ventana["sunset"] - df_ventana["sunrise"]).dt.total_seconds() / 3600
        fig_light = px.line(df_ventana, x="date", y="sunlight_hours", title="Horas de luz solar")
        st.plotly_chart(fig_light, use_container_width=True)

# Evaluar riesgo solar
st.markdown("### ☀️ Calculadora de riesgo solar")
uv = predicciones["Índice UV"]
if uv < 3:
    st.success(f"🟢 Riesgo bajo ({uv}). Puedes exponerte al sol con precaución.")
elif 3 <= uv < 6:
    st.warning(f"🟡 Riesgo moderado ({uv}). Usa protector solar y evita horas punta.")
else:
    st.error(f"🔴 Riesgo alto ({uv}). Evita exposición prolongada entre 12 y 16h.")
