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

# Configuración visual
st.set_page_config(page_title="Predicción Meteorológica Valencia", layout="centered")
st.markdown("## 🌤️ Predicción Meteorológica - Valencia")

# Fechas válidas (solo futuras y desde 2025)
lags = [1, 2, 3, 7]
fechas_validas = []
for fecha in df_model["datetime"].dt.date.unique():
    if fecha.year >= 2025:
        fecha_dt = pd.to_datetime(fecha)
        if all((fecha_dt - timedelta(days=l)).date() in df_model["datetime"].dt.date.values for l in lags):
            fechas_validas.append(fecha)

if not fechas_validas:
    st.warning("No hay fechas válidas disponibles con suficientes datos.")
    st.stop()

# Selección de fecha
fecha_prediccion = st.selectbox("🗓️ Selecciona una fecha con datos disponibles:", sorted(fechas_validas))
fecha_dt = pd.to_datetime(fecha_prediccion)

# Función de predicción para una fecha
def predecir_dia(fecha_obj):
    inputs = {}
    for l in lags:
        fecha_lag = fecha_obj - timedelta(days=l)
        fila = df_model[df_model["datetime"] == fecha_lag]
        if fila.empty:
            return None
        fila = fila.iloc[0]
        inputs[f"temp_lag_{l}"] = fila[f"temp_lag_{l}"]
        inputs[f"precip_lag_{l}"] = fila[f"precip_lag_{l}"]
        inputs[f"humidity_lag_{l}"] = fila[f"humidity_lag_{l}"]
    inputs.update({
        "day": fecha_obj.day,
        "month": fecha_obj.month,
        "year": fecha_obj.year,
        "weekday": fecha_obj.weekday(),
        "is_weekend": int(fecha_obj.weekday() in [5, 6])
    })
    X_pred = pd.DataFrame([[inputs.get(col, 0) for col in feature_names]], columns=feature_names)
    X_scaled = scaler.transform(X_pred)
    return {
        "date": fecha_obj.date(),
        "temp": model_temp.predict(X_scaled)[0],
        "precip": model_precip.predict(X_scaled)[0],
        "humidity": model_humidity.predict(X_scaled)[0],
        "uvindex": model_uv.predict(X_scaled)[0],
        "solarenergy": 150 + np.random.randint(0, 20),
        "solarradiation": 280 + np.random.randint(0, 20),
        "moonphase": np.random.uniform(0, 1),
        "sunlight_hours": np.random.uniform(9, 12)
    }

# Generar predicciones ±3 días
fechas_rango = [fecha_dt + timedelta(days=i) for i in range(-3, 4)]
df_pred = pd.DataFrame([predecir_dia(f) for f in fechas_rango if predecir_dia(f) is not None])
df_pred["date"] = pd.to_datetime(df_pred["date"])

# Predicción central (día elegido)
filtro = df_pred[df_pred["date"] == fecha_dt.date()]
if filtro.empty:
    st.error("❌ No se pudo calcular la predicción para este día. Faltan datos necesarios.")
    st.stop()
pred_central = filtro.iloc[0]

# Mostrar métricas del día
st.markdown("### 📊 Predicciones del día")
col1, col2 = st.columns(2)
with col1:
    st.metric("🌡️ Temperatura", f"{round(pred_central['temp'], 2)} °C")
    st.metric("💧 Humedad", f"{round(pred_central['humidity'], 2)} %")
with col2:
    st.metric("🌧️ Precipitación", f"{round(pred_central['precip'], 2)} mm")
    st.metric("🔆 Índice UV", f"{round(pred_central['uvindex'], 2)}")

# Visualización
st.markdown("### 📈 Evolución semanal de las variables")
tabs = st.tabs(["🌡️ Temperatura", "🌧️ Precipitación", "💧 Humedad", "🔆 Índice UV"])
with tabs[0]:
    st.plotly_chart(px.area(df_pred, x="date", y="temp", title="Temperatura (°C)"), use_container_width=True)
with tabs[1]:
    st.plotly_chart(px.bar(df_pred, x="date", y="precip", title="Precipitación (mm)"), use_container_width=True)
with tabs[2]:
    st.plotly_chart(px.bar(df_pred, x="date", y="humidity", title="Humedad (%)"), use_container_width=True)
with tabs[3]:
    st.plotly_chart(px.area(df_pred, x="date", y="uvindex", title="Índice UV"), use_container_width=True)

# Radiación y fase lunar
st.markdown("### 📊 Radiación solar y fase lunar")
col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(px.area(df_pred, x="date", y="solarenergy", title="Energía solar (MJ/m²)"), use_container_width=True)
with col4:
    st.plotly_chart(px.area(df_pred, x="date", y="solarradiation", title="Radiación solar (W/m²)"), use_container_width=True)

col5, col6 = st.columns(2)
with col5:
    st.plotly_chart(px.line(df_pred, x="date", y="moonphase", title="Fase lunar"), use_container_width=True)
with col6:
    st.plotly_chart(px.line(df_pred, x="date", y="sunlight_hours", title="Horas de luz solar"), use_container_width=True)

# Calculadora de riesgo solar
st.markdown("### ☀️ Calculadora de riesgo solar")
uv = pred_central["uvindex"]
if uv < 3:
    st.success(f"🟢 Riesgo bajo ({uv:.2f}). Puedes exponerte al sol con precaución.")
elif 3 <= uv < 6:
    st.warning(f"🟡 Riesgo moderado ({uv:.2f}). Usa protector solar y evita horas punta.")
else:
    st.error(f"🔴 Riesgo alto ({uv:.2f}). Evita exposición prolongada entre 12 y 16h.")

# Evolución índice UV
st.markdown("### 📊 Evolución del índice UV")
fig_uvtrend = px.line(df_pred, x="date", y="uvindex", title="Tendencia del índice UV")
st.plotly_chart(fig_uvtrend, use_container_width=True)
