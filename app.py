import streamlit as st
import pandas as pd
import joblib
from datetime import date, timedelta
import numpy as np
import plotly.express as px

# Cargar scaler
scaler = joblib.load("scaler.pkl")

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
st.markdown("Esta aplicación predice el clima de Valencia para cualquier fecha futura disponible.")

# === Encontrar fechas válidas para predicción ===
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
    if not faltan_lags and fecha > date.today():
        fechas_validas.append(fecha)

# Controlar si hay fechas válidas
if not fechas_validas:
    st.error("No hay fechas válidas con suficientes datos para predecir.")
    st.stop()

# Selector de fecha segura
date_options = sorted(fechas_validas)
fecha_prediccion = st.selectbox(
    "🗓️ Selecciona una fecha futura con datos disponibles:",
    date_options
)

# Preparar entrada del modelo
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

X_pred = pd.DataFrame([inputs])
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

# Gráfico de barras
st.markdown("### 📈 Visualización de variables")
df_plot = pd.DataFrame(predicciones.items(), columns=["Variable", "Valor"])
fig = px.bar(df_plot, x="Variable", y="Valor", color="Variable",
             title=f"Predicciones para el {fecha_prediccion.strftime('%d/%m/%Y')}")
st.plotly_chart(fig, use_container_width=True)

# Evaluar riesgo solar
st.markdown("### ☀️ Calculadora de riesgo solar")
uv = predicciones["Índice UV"]
if uv < 3:
    st.success(f"🟢 Riesgo bajo ({uv}). Puedes exponerte al sol con precaución.")
elif 3 <= uv < 6:
    st.warning(f"🟡 Riesgo moderado ({uv}). Usa protector solar y evita horas punta.")
else:
    st.error(f"🔴 Riesgo alto ({uv}). Evita exposición prolongada entre 12 y 16h.")
