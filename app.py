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

# Mostrar rango total de fechas disponibles
fechas_disponibles = df_model["datetime"].dt.date.unique()
st.info(f"📅 Datos disponibles desde {min(fechas_disponibles)} hasta {max(fechas_disponibles)}.")

# Buscar fechas válidas
fechas_validas = []
lags = [1, 2, 3, 7]
for fecha in fechas_disponibles:
    fecha_actual = pd.to_datetime(fecha)
    if all((fecha_actual - timedelta(days=l)).date() in fechas_disponibles for l in lags) and fecha > hoy:
        fechas_validas.append(fecha)

# Selector de fecha (solo futuras)
opciones = sorted(fechas_validas)
if not opciones:
    st.warning("No hay fechas futuras disponibles con suficientes datos.")
    st.stop()

fecha_prediccion = st.selectbox("🗓️ Selecciona una fecha con datos disponibles:", opciones)

# Preparar datos
fecha_actual = pd.to_datetime(fecha_prediccion)
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

# Generar predicciones para 7 días consecutivos desde la fecha seleccionada
dias_prediccion = 7
lags = [1, 2, 3, 7]
predicciones_semana = []

for i in range(dias_prediccion):
    fecha_dia = fecha_actual + timedelta(days=i)
    entrada = {}
    datos_ok = True

    for l in lags:
        fecha_lag = fecha_dia - timedelta(days=l)
        fila = df_model[df_model["datetime"].dt.date == fecha_lag.date()]
        if fila.empty:
            datos_ok = False
            break
        fila = fila.iloc[0]
        entrada[f"temp_lag_{l}"] = fila[f"temp_lag_{l}"]
        entrada[f"precip_lag_{l}"] = fila[f"precip_lag_{l}"]
        entrada[f"humidity_lag_{l}"] = fila[f"humidity_lag_{l}"]

    if datos_ok:
        entrada.update({
            "day": fecha_dia.day,
            "month": fecha_dia.month,
            "year": fecha_dia.year,
            "weekday": fecha_dia.weekday(),
            "is_weekend": int(fecha_dia.weekday() in [5, 6])
        })
        X_pred_semana = pd.DataFrame([[entrada.get(col, 0) for col in feature_names]], columns=feature_names)
        X_scaled_semana = scaler.transform(X_pred_semana)

        predicciones_semana.append({
            "date": fecha_dia.date(),
            "temp": model_temp.predict(X_scaled_semana)[0],
            "precip": model_precip.predict(X_scaled_semana)[0],
            "humidity": model_humidity.predict(X_scaled_semana)[0],
            "uvindex": model_uv.predict(X_scaled_semana)[0]
        })

df_pred = pd.DataFrame(predicciones_semana)

# Mostrar gráficos con valores predichos
st.markdown("### 📈 Evolución semanal de las variables")
tabs = st.tabs(["🌡️ Temperatura", "🌧️ Precipitación", "💧 Humedad", "🔆 Índice UV"])

with tabs[0]:
    fig_temp = px.area(df_pred, x="date", y="temp", title="Temperatura (°C)")
    st.plotly_chart(fig_temp, use_container_width=True)

with tabs[1]:
    fig_precip = px.bar(df_pred, x="date", y="precip", title="Precipitación (mm)")
    st.plotly_chart(fig_precip, use_container_width=True)

with tabs[2]:
    fig_hum = px.bar(df_pred, x="date", y="humidity", title="Humedad (%)")
    st.plotly_chart(fig_hum, use_container_width=True)

with tabs[3]:
    fig_uv = px.area(df_pred, x="date", y="uvindex", title="Índice UV")
    st.plotly_chart(fig_uv, use_container_width=True)

# Dashboard avanzado
with st.expander("📊 Dashboard avanzado"):
    st.subheader("🔆 Radiación y Energía Solar")
    fig_sol = px.area(df_ventana, x="date", y=["solarenergy", "solarradiation"],
                      title="Radiación y energía solar", labels={"value": "Intensidad", "variable": "Variable"})
    st.plotly_chart(fig_sol, use_container_width=True)

    st.subheader("🌙 Fase lunar")
    fig_luna = px.area(df_ventana, x="date", y="moonphase", title="Fase lunar",
                       labels={"moonphase": "Fase (0=nueva, 1=llena)"})
    st.plotly_chart(fig_luna, use_container_width=True)

# Evaluar riesgo solar
st.markdown("### ☀️ Calculadora de riesgo solar")
uv = predicciones["Índice UV"]
if uv < 3:
    st.success(f"🟢 Riesgo bajo ({uv}). Puedes exponerte al sol con precaución.")
elif 3 <= uv < 6:
    st.warning(f"🟡 Riesgo moderado ({uv}). Usa protector solar y evita horas punta.")
else:
    st.error(f"🔴 Riesgo alto ({uv}). Evita exposición prolongada entre 12 y 16h.")
