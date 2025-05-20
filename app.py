import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import plotly.express as px

# Simulación de "hoy"
hoy = datetime(2025, 1, 1).date()

# Cargar recursos
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

# Configuración de la app
st.set_page_config(page_title="Predicción Meteorológica Valencia", layout="centered")
st.markdown("## 🌤️ Predicción Meteorológica - Valencia")
st.markdown("Esta aplicación predice el clima de Valencia para cualquier fecha con datos disponibles.")

# Rango de fechas
fechas_disponibles = df_model["date"].unique()
st.info(f"📅 Datos disponibles desde {min(fechas_disponibles)} hasta {max(fechas_disponibles)}.")

# Obtener fechas válidas con lags disponibles
lags = [1, 2, 3, 7]
fechas_validas = []
for fecha in fechas_disponibles:
    fecha_dt = pd.to_datetime(fecha)
    if fecha > hoy and all((fecha_dt - timedelta(days=l)).date() in fechas_disponibles for l in lags):
        fechas_validas.append(fecha)

if not fechas_validas:
    st.error("No hay fechas válidas disponibles con suficientes datos.")
    st.stop()

# Selección de fecha
fecha_pred = st.selectbox("🗓️ Selecciona una fecha con datos disponibles:", sorted(fechas_validas))
fecha_dt = pd.to_datetime(fecha_pred)

# Generar predicciones para 7 días desde la fecha seleccionada
predicciones = []
for i in range(7):
    fecha_actual = fecha_dt + timedelta(days=i)
    datos_ok = True
    entrada = {}

    for l in lags:
        fecha_lag = fecha_actual - timedelta(days=l)
        fila = df_model[df_model["date"] == fecha_lag.date()]
        if fila.empty:
            datos_ok = False
            break
        fila = fila.iloc[0]
        entrada[f"temp_lag_{l}"] = fila[f"temp_lag_{l}"]
        entrada[f"precip_lag_{l}"] = fila[f"precip_lag_{l}"]
        entrada[f"humidity_lag_{l}"] = fila[f"humidity_lag_{l}"]

    if datos_ok:
        entrada["day"] = fecha_actual.day
        entrada["month"] = fecha_actual.month
        entrada["year"] = fecha_actual.year
        entrada["weekday"] = fecha_actual.weekday()
        entrada["is_weekend"] = int(fecha_actual.weekday() in [5, 6])

        X = pd.DataFrame([[entrada.get(col, 0) for col in feature_names]], columns=feature_names)
        X_scaled = scaler.transform(X)

        predicciones.append({
            "date": fecha_actual.date(),
            "temp": model_temp.predict(X_scaled)[0],
            "precip": model_precip.predict(X_scaled)[0],
            "humidity": model_humidity.predict(X_scaled)[0],
            "uvindex": model_uv.predict(X_scaled)[0]
        })

# Convertir predicciones en DataFrame
df_pred = pd.DataFrame(predicciones)

# Mostrar predicción puntual
st.markdown("### 📊 Predicciones del día")
fila_hoy = df_pred[df_pred["date"] == fecha_pred].iloc[0]
col1, col2 = st.columns(2)
with col1:
    st.metric("🌡️ Temperatura", f"{round(fila_hoy['temp'], 2)} °C")
    st.metric("💧 Humedad", f"{round(fila_hoy['humidity'], 2)} %")
with col2:
    st.metric("🌧️ Precipitación", f"{round(fila_hoy['precip'], 2)} mm")
    st.metric("🔆 Índice UV", f"{round(fila_hoy['uvindex'], 2)}")

# Gráficos semanales con predicciones
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

# Evaluación de riesgo UV
st.markdown("### ☀️ Calculadora de riesgo solar")
uv = fila_hoy["uvindex"]
if uv < 3:
    st.success(f"🟢 Riesgo bajo ({round(uv, 2)}). Puedes exponerte al sol con precaución.")
elif 3 <= uv < 6:
    st.warning(f"🟡 Riesgo moderado ({round(uv, 2)}). Usa protector solar y evita horas punta.")
else:
    st.error(f"🔴 Riesgo alto ({round(uv, 2)}). Evita exposición prolongada entre 12 y 16h.")
