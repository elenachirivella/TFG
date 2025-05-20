import streamlit as st
import pandas as pd
import joblib
from datetime import timedelta, datetime
import numpy as np
import plotly.express as px

# ⚠️ Simulación de "hoy" para el contexto del TFG
hoy = datetime(2025, 1, 1).date()

# Cargar scaler, columnas y modelos
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
model_temp = joblib.load("rf_temp_futuro_30.pkl")
model_precip = joblib.load("rf_precip_futuro_30.pkl")
model_humidity = joblib.load("rf_humidity_futuro_30.pkl")
model_uv = joblib.load("rf_uvindex_futuro_30.pkl")

# Cargar dataset
df_model = pd.read_csv("df_model_final.csv")
df_model["datetime"] = pd.to_datetime(df_model["datetime"], errors='coerce')
df_model = df_model.dropna(subset=["datetime"])
df_model["date"] = df_model["datetime"].dt.date

# Obtener fechas futuras válidas
lags = [1, 2, 3, 7]
fechas_disponibles = df_model["date"].unique()
fechas_validas = []
for fecha in fechas_disponibles:
    fecha_actual = pd.to_datetime(fecha)
    if all((fecha_actual - timedelta(days=l)).date() in fechas_disponibles for l in lags) and fecha > hoy:
        fechas_validas.append(fecha)

# Selector de fecha
st.set_page_config(page_title="Predicción Meteorológica Valencia", layout="centered")
st.markdown("## 🌤️ Predicción Meteorológica - Valencia")
st.info(f"📅 Datos disponibles desde {min(fechas_disponibles)} hasta {max(fechas_disponibles)}.")

if not fechas_validas:
    st.warning("No hay fechas futuras disponibles con suficientes datos.")
    st.stop()

fecha_seleccionada = st.selectbox("🗓️ Selecciona una fecha con datos disponibles:", sorted(fechas_validas))

# Generar predicciones para 7 días desde la fecha seleccionada
predicciones_semana = []
for i in range(7):
    fecha_actual = pd.to_datetime(fecha_seleccionada) + timedelta(days=i)
    fecha_actual_date = fecha_actual.date()

    inputs = {}
    datos_ok = True
    for l in lags:
        fecha_lag = fecha_actual - timedelta(days=l)
        fila = df_model[df_model["date"] == fecha_lag.date()]
        if fila.empty:
            datos_ok = False
            break
        fila = fila.iloc[0]
        inputs[f"temp_lag_{l}"] = fila[f"temp_lag_{l}"]
        inputs[f"precip_lag_{l}"] = fila[f"precip_lag_{l}"]
        inputs[f"humidity_lag_{l}"] = fila[f"humidity_lag_{l}"]

    if not datos_ok:
        continue

    inputs.update({
        "day": fecha_actual.day,
        "month": fecha_actual.month,
        "year": fecha_actual.year,
        "weekday": fecha_actual.weekday(),
        "is_weekend": int(fecha_actual.weekday() in [5, 6])
    })

    X_pred = pd.DataFrame([[inputs.get(col, 0) for col in feature_names]], columns=feature_names)
    X_scaled = scaler.transform(X_pred)

    predicciones_semana.append({
        "date": fecha_actual_date,
        "temp": model_temp.predict(X_scaled)[0],
        "precip": model_precip.predict(X_scaled)[0],
        "humidity": model_humidity.predict(X_scaled)[0],
        "uvindex": model_uv.predict(X_scaled)[0]
    })

df_pred = pd.DataFrame(predicciones_semana)

# Mostrar predicción puntual del día seleccionado
st.markdown("### 📊 Predicciones del día")
pred_hoy = df_pred[df_pred["date"] == fecha_seleccionada].iloc[0]
col1, col2 = st.columns(2)
with col1:
    st.metric("🌡️ Temperatura", f"{round(pred_hoy['temp'],2)} °C")
    st.metric("💧 Humedad", f"{round(pred_hoy['humidity'],2)} %")
with col2:
    st.metric("🌧️ Precipitación", f"{round(pred_hoy['precip'],2)} mm")
    st.metric("🔆 Índice UV", f"{round(pred_hoy['uvindex'],2)}")

# Gráficos de evolución con los valores predichos
st.markdown("### 📈 Evolución semanal de las variables")
tabs = st.tabs(["🌡️ Temperatura", "🌧️ Precipitación", "💧 Humedad", "🔆 Índice UV"])

with tabs[0]:
    fig = px.area(df_pred, x="date", y="temp", title="Temperatura (°C)")
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    fig = px.bar(df_pred, x="date", y="precip", title="Precipitación (mm)")
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    fig = px.bar(df_pred, x="date", y="humidity", title="Humedad (%)")
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    fig = px.area(df_pred, x="date", y="uvindex", title="Índice UV")
    st.plotly_chart(fig, use_container_width=True)

# Evaluar riesgo solar para el día seleccionado
st.markdown("### ☀️ Calculadora de riesgo solar")
uv = pred_hoy["uvindex"]
if uv < 3:
    st.success(f"🟢 Riesgo bajo ({round(uv,2)}). Puedes exponerte al sol con precaución.")
elif 3 <= uv < 6:
    st.warning(f"🟡 Riesgo moderado ({round(uv,2)}). Usa protector solar y evita horas punta.")
else:
    st.error(f"🔴 Riesgo alto ({round(uv,2)}). Evita exposición prolongada entre 12 y 16h.")

