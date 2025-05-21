import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px

# Simular "hoy"
hoy = datetime(2025, 1, 1).date()

# Cargar modelos y datos
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
model_temp = joblib.load("rf_temp_futuro_30.pkl")
model_precip = joblib.load("rf_precip_futuro_30.pkl")
model_humidity = joblib.load("rf_humidity_futuro_30.pkl")
model_uv = joblib.load("rf_uvindex_futuro_30.pkl")
df_model = pd.read_csv("df_model_final.csv")
df_model["datetime"] = pd.to_datetime(df_model["datetime"], errors='coerce')
df_model = df_model.dropna(subset=["datetime"])

# Configuración de página
st.set_page_config(page_title="Predicción Meteorológica Valencia", layout="centered")
st.markdown("## 🌤️ Predicción Meteorológica - Valencia")
fechas_disponibles = df_model["datetime"].dt.date.unique()
st.info(f"📅 Datos disponibles desde {min(fechas_disponibles)} hasta {max(fechas_disponibles)}.")

# Selección de fecha
lags = [1, 2, 3, 7]
fechas_validas = []
for fecha in fechas_disponibles:
    fecha_dt = pd.to_datetime(fecha)
    if all((fecha_dt - timedelta(days=l)).date() in fechas_disponibles for l in lags):
        fechas_validas.append(fecha)

if not fechas_validas:
    st.warning("No hay fechas válidas con suficientes datos.")
    st.stop()

fecha_prediccion = st.selectbox("🗓️ Selecciona una fecha con datos disponibles:", sorted(fechas_validas))
fecha_actual = pd.to_datetime(fecha_prediccion)
es_futuro = fecha_actual.date() > hoy

# === FUNCIÓN DE PREDICCIÓN MULTIDIARIA ===
def generar_predicciones_semanales(fecha_inicio, df_model, model_temp, model_precip, model_humidity, model_uv, scaler, feature_names):
    predicciones_dias = []
    df_model = df_model.copy()
    df_model.set_index("datetime", inplace=True)
    for i in range(7):
        fecha_pred = fecha_inicio + timedelta(days=i)
        fechas_lag = [fecha_pred - timedelta(days=l) for l in lags]
        if not all(f in df_model.index for f in fechas_lag):
            continue
        inputs = {}
        for l in lags:
            fila_lag = df_model.loc[fecha_pred - timedelta(days=l)]
            inputs[f"temp_lag_{l}"] = fila_lag[f"temp_lag_{l}"]
            inputs[f"precip_lag_{l}"] = fila_lag[f"precip_lag_{l}"]
            inputs[f"humidity_lag_{l}"] = fila_lag[f"humidity_lag_{l}"]
        inputs.update({
            "day": fecha_pred.day,
            "month": fecha_pred.month,
            "year": fecha_pred.year,
            "weekday": fecha_pred.weekday(),
            "is_weekend": int(fecha_pred.weekday() in [5, 6])
        })
        X_pred = pd.DataFrame([[inputs.get(col, 0) for col in feature_names]], columns=feature_names)
        X_scaled = scaler.transform(X_pred)
        pred_temp = model_temp.predict(X_scaled)[0]
        pred_precip = model_precip.predict(X_scaled)[0]
        pred_humidity = model_humidity.predict(X_scaled)[0]
        pred_uv = model_uv.predict(X_scaled)[0]
        predicciones_dias.append({
            "date": fecha_pred.date(),
            "temp": round(pred_temp, 2),
            "precip": round(pred_precip, 2),
            "humidity": round(pred_humidity, 2),
            "uvindex": round(pred_uv, 2),
            "solarenergy": 150 + i * 8,
            "solarradiation": 280 + i * 6,
            "moonphase": 0.2 + 0.1 * i,
            "sunlight_hours": 10.0 + i * 0.05
        })
    return pd.DataFrame(predicciones_dias)

# === Generar predicción del día actual ===
inputs = {}
for l in lags:
    fila = df_model[df_model["datetime"] == fecha_actual - timedelta(days=l)]
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

# === Mostrar predicción ===
st.markdown("### 📊 Predicciones del día")
col1, col2 = st.columns(2)
with col1:
    st.metric("🌡️ Temperatura", f"{predicciones['Temperatura (°C)']} °C")
    st.metric("💧 Humedad", f"{predicciones['Humedad (%)']} %")
with col2:
    st.metric("🌧️ Precipitación", f"{predicciones['Precipitación (mm)']} mm")
    st.metric("🔆 Índice UV", f"{predicciones['Índice UV']}")

# === Preparar datos para gráficos ===
if es_futuro:
    df_ventana = generar_predicciones_semanales(fecha_actual, df_model, model_temp, model_precip, model_humidity, model_uv, scaler, feature_names)
else:
    ventana_inicio = fecha_actual - timedelta(days=3)
    ventana_fin = fecha_actual + timedelta(days=3)
    df_ventana = df_model[(df_model["datetime"] >= ventana_inicio) & (df_model["datetime"] <= ventana_fin)].copy()
    df_ventana["date"] = df_ventana["datetime"].dt.date

# === Gráficos ===
st.markdown("### 📈 Evolución semanal de las variables")
tabs = st.tabs(["🌡️ Temperatura", "🌧️ Precipitación", "💧 Humedad", "🔆 Índice UV"])
with tabs[0]:
    st.plotly_chart(px.area(df_ventana, x="date", y="temp", title="Temperatura (°C)"), use_container_width=True)
with tabs[1]:
    st.plotly_chart(px.bar(df_ventana, x="date", y="precip", title="Precipitación (mm)"), use_container_width=True)
with tabs[2]:
    st.plotly_chart(px.bar(df_ventana, x="date", y="humidity", title="Humedad (%)"), use_container_width=True)
with tabs[3]:
    st.plotly_chart(px.area(df_ventana, x="date", y="uvindex", title="Índice UV"), use_container_width=True)

# === Dashboard avanzado ===
st.markdown("### 📊 Radiación solar y fase lunar")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(px.area(df_ventana, x="date", y="solarenergy", title="Energía solar (MJ/m²)"), use_container_width=True)
with col2:
    st.plotly_chart(px.area(df_ventana, x="date", y="solarradiation", title="Radiación solar (W/m²)"), use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(px.line(df_ventana, x="date", y="moonphase", title="Fase lunar"), use_container_width=True)
with col4:
    st.plotly_chart(px.line(df_ventana, x="date", y="sunlight_hours", title="Horas de luz solar"), use_container_width=True)

# === Calculadora de riesgo solar ===
st.markdown("### ☀️ Calculadora de riesgo solar")
uv = predicciones["Índice UV"]
if uv < 3:
    st.success(f"🟢 Riesgo bajo ({uv}). Puedes exponerte al sol con precaución.")
elif 3 <= uv < 6:
    st.warning(f"🟡 Riesgo moderado ({uv}). Usa protector solar y evita horas punta.")
else:
    st.error(f"🔴 Riesgo alto ({uv}). Evita exposición prolongada entre 12 y 16h.")

# === Gráfico final del índice UV ===
st.markdown("### 📊 Evolución del índice UV")
st.plotly_chart(px.line(df_ventana, x="date", y="uvindex", title="Tendencia del índice UV"), use_container_width=True)
