import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px

# ⚠️ Fecha fija para simular "hoy"
hoy = datetime(2025, 1, 1).date()

# Cargar modelos y escalador
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
model_temp = joblib.load("rf_temp_futuro_30.pkl")
model_precip = joblib.load("rf_precip_futuro_30.pkl")
model_humidity = joblib.load("rf_humidity_futuro_30.pkl")
model_uv = joblib.load("rf_uvindex_futuro_30.pkl")

# Cargar dataset
df_model = pd.read_csv("df_model_final.csv")
df_model["datetime"] = pd.to_datetime(df_model["datetime"], errors="coerce")
df_model = df_model.dropna(subset=["datetime"])
fechas_disponibles = df_model["datetime"].dt.date.unique()

# Filtrar solo fechas válidas futuras
lags = [1, 2, 3, 7]
fechas_validas = []
for fecha in fechas_disponibles:
    fecha_actual = pd.to_datetime(fecha)
    if fecha > hoy and all((fecha_actual - timedelta(days=l)).date() in fechas_disponibles for l in lags):
        fechas_validas.append(fecha)

# Título y selector
st.set_page_config(page_title="Predicción Meteorológica Valencia", layout="centered")
st.markdown("## 🌤️ Predicción Meteorológica - Valencia")
st.info(f"📅 Datos disponibles desde {min(fechas_disponibles)} hasta {max(fechas_disponibles)}.")

if not fechas_validas:
    st.warning("No hay fechas futuras disponibles con suficientes datos.")
    st.stop()

fecha_prediccion = st.selectbox("🗓️ Selecciona una fecha con datos disponibles:", sorted(fechas_validas))

# Crear entrada para predicción
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
    "Temperatura (°C)": round(pred_temp, 2),
    "Precipitación (mm)": round(pred_precip, 2),
    "Humedad (%)": round(pred_humidity, 2),
    "Índice UV": round(pred_uv, 2)
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

# Ventana de ±3 días
ventana_inicio = fecha_actual - timedelta(days=3)
ventana_fin = fecha_actual + timedelta(days=3)
df_ventana = df_model[(df_model["datetime"] >= ventana_inicio) & (df_model["datetime"] <= ventana_fin)].copy()
df_ventana["date"] = df_ventana["datetime"].dt.date

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

# Dashboard visual con explicación
st.markdown("### 📊 Radiación solar y fase lunar")

# Radiación y energía solar
df_radiacion = df_ventana[["date", "solarenergy", "solarradiation"]].copy()
df_radiacion = df_radiacion.melt(id_vars="date", var_name="variable", value_name="valor")
fig_rad = px.area(df_radiacion, x="date", y="valor", color="variable",
                  title="Radiación y energía solar",
                  labels={"valor": "Intensidad", "date": "Fecha", "variable": "Variable"},
                  color_discrete_sequence=["#4472C4", "#A9D18E"])
st.plotly_chart(fig_rad, use_container_width=True)
st.caption("🔎 Días con menor radiación solar generan también menor acumulación de energía. Observa si hay nubes o lluvias coincidentes.")

# Fase lunar y horas de luz
df_luna = df_ventana[["date", "moonphase"]].copy()
df_luna["sunrise"] = pd.to_datetime(df_ventana["sunrise"], errors='coerce')
df_luna["sunset"] = pd.to_datetime(df_ventana["sunset"], errors='coerce')
df_luna["sunlight_hours"] = (df_luna["sunset"] - df_luna["sunrise"]).dt.total_seconds() / 3600
df_luna = df_luna[["date", "moonphase", "sunlight_hours"]]
df_luna_melt = df_luna.melt(id_vars="date", var_name="variable", value_name="valor")
fig_luna = px.bar(df_luna_melt, x="date", y="valor", color="variable", barmode="group",
                  title="Fase lunar y horas de luz solar",
                  labels={"valor": "Valor", "variable": "Variable"},
                  color_discrete_sequence=["#558ED5", "#ED7D31"])
st.plotly_chart(fig_luna, use_container_width=True)
st.caption("🌙 La fase lunar cambia progresivamente mientras que las horas de luz se mantienen estables cerca del solsticio.")

# Evaluación del riesgo solar con mini visual extra
st.markdown("### ☀️ Calculadora de riesgo solar")
uv = predicciones["Índice UV"]
if uv < 3:
    st.success(f"🟢 Riesgo bajo ({uv}). Puedes exponerte al sol con precaución.")
elif 3 <= uv < 6:
    st.warning(f"🟡 Riesgo moderado ({uv}). Usa protector solar y evita horas punta.")
else:
    st.error(f"🔴 Riesgo alto ({uv}). Evita exposición prolongada entre 12 y 16h.")

# Mini gráfico del UV con color
st.markdown("#### 📊 Evolución del índice UV")
fig_uv_trend = px.line(df_ventana, x="date", y="uvindex", markers=True, title="Tendencia del índice UV")
st.plotly_chart(fig_uv_trend, use_container_width=True)
