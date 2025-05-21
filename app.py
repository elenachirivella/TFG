import streamlit as st
import pandas as pd
import joblib
from datetime import timedelta, datetime
import plotly.express as px

# Simular "hoy"
hoy = datetime(2025, 1, 1).date()

# Cargar modelos y configuraciones
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

# Cabecera
st.set_page_config(page_title="Predicción Meteorológica Valencia", layout="centered")
st.markdown("## 🌤️ Predicción Meteorológica - Valencia")
st.markdown("Esta aplicación predice el clima de Valencia para cualquier fecha futura con datos disponibles.")
st.info(f"📅 Datos disponibles desde {min(df_model['date'])} hasta {max(df_model['date'])}.")

# Selección de fechas válidas con lags disponibles
lags = [1, 2, 3, 7]
fechas_disponibles = df_model["date"].unique()
fechas_validas = []

for fecha in fechas_disponibles:
    fecha_dt = pd.to_datetime(fecha)
    if fecha > hoy and all((fecha_dt - timedelta(days=l)).date() in fechas_disponibles for l in lags):
        fechas_validas.append(fecha)

if not fechas_validas:
    st.warning("No hay fechas futuras con datos suficientes.")
    st.stop()

# Seleccionar una fecha válida
fecha_pred = st.selectbox("🗓️ Selecciona una fecha con datos disponibles:", sorted(fechas_validas))
fecha_dt = pd.to_datetime(fecha_pred)

# Preparar datos para predicción
inputs = {}
for l in lags:
    fecha_lag = fecha_dt - timedelta(days=l)
    fila = df_model[df_model["date"] == fecha_lag.date()]
    if fila.empty:
        st.error(f"Faltan datos para la fecha: {fecha_lag.date()}.")
        st.stop()
    fila = fila.iloc[0]
    inputs[f"temp_lag_{l}"] = fila[f"temp_lag_{l}"]
    inputs[f"precip_lag_{l}"] = fila[f"precip_lag_{l}"]
    inputs[f"humidity_lag_{l}"] = fila[f"humidity_lag_{l}"]

inputs.update({
    "day": fecha_dt.day,
    "month": fecha_dt.month,
    "year": fecha_dt.year,
    "weekday": fecha_dt.weekday(),
    "is_weekend": int(fecha_dt.weekday() in [5, 6])
})

X_pred = pd.DataFrame([[inputs.get(col, 0) for col in feature_names]], columns=feature_names)
X_scaled = scaler.transform(X_pred)

# Predicciones
predicciones = {
    "Temperatura (°C)": round(model_temp.predict(X_scaled)[0], 2),
    "Precipitación (mm)": round(model_precip.predict(X_scaled)[0], 2),
    "Humedad (%)": round(model_humidity.predict(X_scaled)[0], 2),
    "Índice UV": round(model_uv.predict(X_scaled)[0], 2)
}

# Mostrar predicción
st.markdown("### 📊 Predicciones del día")
col1, col2 = st.columns(2)
with col1:
    st.metric("🌡️ Temperatura", f"{predicciones['Temperatura (°C)']} °C")
    st.metric("💧 Humedad", f"{predicciones['Humedad (%)']} %")
with col2:
    st.metric("🌧️ Precipitación", f"{predicciones['Precipitación (mm)']} mm")
    st.metric("🔆 Índice UV", f"{predicciones['Índice UV']}")

# Gráficos semanales (±3 días)
ventana_inicio = fecha_dt - timedelta(days=3)
ventana_fin = fecha_dt + timedelta(days=3)
df_ventana = df_model[(df_model["datetime"] >= ventana_inicio) & (df_model["datetime"] <= ventana_fin)].copy()
df_ventana["date"] = df_ventana["datetime"].dt.date

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

# Dashboard avanzado (sin humedad relativa)
with st.expander("📊 Dashboard avanzado"):
    st.subheader("🔆 Radiación y Energía Solar")
    col1, col2 = st.columns(2)
    with col1:
        fig_solar_energy = px.area(df_ventana, x="date", y="solarenergy", title="Energía solar (MJ/m²)")
        st.plotly_chart(fig_solar_energy, use_container_width=True)
    with col2:
        fig_solar_rad = px.area(df_ventana, x="date", y="solarradiation", title="Radiación solar (W/m²)")
        st.plotly_chart(fig_solar_rad, use_container_width=True)

    st.subheader("🌙 Fase lunar y duración del día")
    col3, col4 = st.columns(2)
    with col3:
        fig_moon = px.line(df_ventana, x="date", y="moonphase", title="Fase lunar")
        st.plotly_chart(fig_moon, use_container_width=True)
    with col4:
        df_ventana["sunrise"] = pd.to_datetime(df_ventana["sunrise"], errors='coerce')
        df_ventana["sunset"] = pd.to_datetime(df_ventana["sunset"], errors='coerce')
        df_ventana["sunlight_hours"] = (df_ventana["sunset"] - df_ventana["sunrise"]).dt.total_seconds() / 3600
        fig_light = px.line(df_ventana, x="date", y="sunlight_hours", title="Horas de luz solar")
        st.plotly_chart(fig_light, use_container_width=True)

# Evaluación de riesgo solar
st.markdown("### ☀️ Calculadora de riesgo solar")
uv = predicciones["Índice UV"]
if uv < 3:
    st.success(f"🟢 Riesgo bajo ({uv}). Puedes exponerte al sol con precaución.")
elif 3 <= uv < 6:
    st.warning(f"🟡 Riesgo moderado ({uv}). Usa protector solar y evita horas punta.")
else:
    st.error(f"🔴 Riesgo alto ({uv}). Evita exposición prolongada entre 12 y 16h.")
