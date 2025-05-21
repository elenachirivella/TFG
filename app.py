import streamlit as st
import pandas as pd
import joblib
from datetime import timedelta, datetime
import plotly.express as px

# Simular fecha de hoy
hoy = datetime(2025, 1, 1).date()

# Cargar recursos
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
model_temp = joblib.load("rf_temp_futuro_30.pkl")
model_precip = joblib.load("rf_precip_futuro_30.pkl")
model_humidity = joblib.load("rf_humidity_futuro_30.pkl")
model_uv = joblib.load("rf_uvindex_futuro_30.pkl")

# Dataset
df_model = pd.read_csv("df_model_final.csv")
df_model["datetime"] = pd.to_datetime(df_model["datetime"], errors='coerce')
df_model = df_model.dropna(subset=["datetime"])
df_model["date"] = df_model["datetime"].dt.date

# Config visual
st.set_page_config(page_title="Predicción Meteorológica Valencia", layout="centered")
st.markdown("## 🌤️ Predicción Meteorológica - Valencia")
st.markdown("Esta aplicación predice el clima de Valencia para cualquier fecha futura con datos disponibles.")
st.info(f"📅 Datos disponibles desde {min(df_model['date'])} hasta {max(df_model['date'])}.")

# Fechas válidas con lags
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

# Selector
fecha_pred = st.selectbox(
    "🗓️ Selecciona una fecha con datos disponibles:",
    sorted(fechas_validas),
    index=len(fechas_validas)-1
)
fecha_dt = pd.to_datetime(fecha_pred)

# Predicción para 7 días
predicciones = []
for i in range(7):
    fecha_actual = fecha_dt + timedelta(days=i)
    entrada = {}
    datos_ok = True

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
        entrada.update({
            "day": fecha_actual.day,
            "month": fecha_actual.month,
            "year": fecha_actual.year,
            "weekday": fecha_actual.weekday(),
            "is_weekend": int(fecha_actual.weekday() in [5, 6])
        })

        X = pd.DataFrame([[entrada.get(col, 0) for col in feature_names]], columns=feature_names)
        X_scaled = scaler.transform(X)

        predicciones.append({
            "date": fecha_actual.date(),
            "temp": model_temp.predict(X_scaled)[0],
            "precip": model_precip.predict(X_scaled)[0],
            "humidity": model_humidity.predict(X_scaled)[0],
            "uvindex": model_uv.predict(X_scaled)[0]
        })

df_pred = pd.DataFrame(predicciones)
fila_hoy = df_pred[df_pred["date"] == fecha_pred].iloc[0]

# Predicción del día
st.markdown("### 📊 Predicciones del día")
col1, col2 = st.columns(2)
with col1:
    st.metric("🌡️ Temperatura", f"{round(fila_hoy['temp'], 2)} °C")
    st.metric("💧 Humedad", f"{round(fila_hoy['humidity'], 2)} %")
with col2:
    st.metric("🌧️ Precipitación", f"{round(fila_hoy['precip'], 2)} mm")
    st.metric("🔆 Índice UV", f"{round(fila_hoy['uvindex'], 2)}")

# Gráficos semanales
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

# Dashboard visual (sin expander)
st.markdown("### 📊 Radiación solar y fase lunar")
df_ventana = df_model[df_model["date"].isin(df_pred["date"])].copy()

col5, col6 = st.columns(2)
with col5:
    fig1 = px.area(df_ventana, x="date", y="solarenergy", title="Energía solar (MJ/m²)")
    st.plotly_chart(fig1, use_container_width=True)
with col6:
    fig2 = px.area(df_ventana, x="date", y="solarradiation", title="Radiación solar (W/m²)")
    st.plotly_chart(fig2, use_container_width=True)

col7, col8 = st.columns(2)
with col7:
    fig3 = px.line(df_ventana, x="date", y="moonphase", title="Fase lunar")
    st.plotly_chart(fig3, use_container_width=True)
with col8:
    df_ventana["sunrise"] = pd.to_datetime(df_ventana["sunrise"], errors='coerce')
    df_ventana["sunset"] = pd.to_datetime(df_ventana["sunset"], errors='coerce')
    df_ventana["sunlight_hours"] = (df_ventana["sunset"] - df_ventana["sunrise"]).dt.total_seconds() / 3600
    fig4 = px.line(df_ventana, x="date", y="sunlight_hours", title="Horas de luz solar")
    st.plotly_chart(fig4, use_container_width=True)

# Calculadora de riesgo solar mejorada
st.markdown("### ☀️ Calculadora de riesgo solar")

uv = fila_hoy["uvindex"]
if uv < 3:
    st.success(f"🟢 Riesgo bajo ({uv}). Puedes exponerte al sol con precaución.")
elif 3 <= uv < 6:
    st.warning(f"🟡 Riesgo moderado ({uv}). Usa protector solar y evita horas punta.")
else:
    st.error(f"🔴 Riesgo alto ({uv}). Evita exposición prolongada entre 12 y 16h.")

# Nuevo gráfico de riesgo UV por colores
st.markdown("#### 🔍 Índice UV durante la semana")
df_pred["riesgo"] = pd.cut(df_pred["uvindex"],
                           bins=[0, 3, 6, 11],
                           labels=["Bajo", "Moderado", "Alto"],
                           include_lowest=True)
colores = {"Bajo": "green", "Moderado": "orange", "Alto": "red"}

fig_uv_bar = px.bar(df_pred, x="date", y="uvindex", color="riesgo",
                    color_discrete_map=colores,
                    title="Índice UV diario con niveles de riesgo",
                    labels={"uvindex": "Índice UV", "date": "Fecha"})

st.plotly_chart(fig_uv_bar, use_container_width=True)
