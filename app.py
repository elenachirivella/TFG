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

# Configuración de página (responsive)
st.set_page_config(page_title="Predicción Meteorológica Valencia", layout="wide")
st.markdown("""
<style>
@media screen and (max-width: 768px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    h1, h2, h3 {
        font-size: 1.4rem !important;
    }
    .stMetric {
        font-size: 1.4rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🌤️ Predicción Meteorológica - Valencia")
fechas_disponibles = df_model["datetime"].dt.date.unique()
st.info(f"🗓️ Datos disponibles desde {min(fechas_disponibles)} hasta {max(fechas_disponibles)}.")

# Validar fechas futuras y completas
lags = [1, 2, 3, 7]
fechas_validas = []
for fecha in fechas_disponibles:
    if fecha > hoy and fecha.year in [2025, 2026]:
        fecha_dt = pd.to_datetime(fecha)
        if all((fecha_dt - timedelta(days=l)).date() in fechas_disponibles for l in lags):
            fechas_validas.append(fecha)

if not fechas_validas:
    st.warning("No hay fechas futuras válidas con suficientes datos.")
    st.stop()

fecha_prediccion = st.selectbox("🗓️ Selecciona una fecha FUTURA con datos disponibles:", sorted(fechas_validas))
fecha_actual = pd.to_datetime(fecha_prediccion)

# Función de predicción múltiple
def generar_predicciones(fecha_inicio):
    predicciones = []
    df_indexed = df_model.set_index("datetime")
    for i in range(7):
        fecha_pred = fecha_inicio + timedelta(days=i)
        fechas_lag = [fecha_pred - timedelta(days=l) for l in lags]
        if not all(f in df_indexed.index for f in fechas_lag):
            continue
        datos = {}
        for l in lags:
            fila_lag = df_indexed.loc[fecha_pred - timedelta(days=l)]
            datos[f"temp_lag_{l}"] = fila_lag[f"temp_lag_{l}"]
            datos[f"precip_lag_{l}"] = fila_lag[f"precip_lag_{l}"]
            datos[f"humidity_lag_{l}"] = fila_lag[f"humidity_lag_{l}"]
        datos.update({
            "day": fecha_pred.day,
            "month": fecha_pred.month,
            "year": fecha_pred.year,
            "weekday": fecha_pred.weekday(),
            "is_weekend": int(fecha_pred.weekday() in [5, 6])
        })
        X_pred = pd.DataFrame([[datos.get(col, 0) for col in feature_names]], columns=feature_names)
        X_scaled = scaler.transform(X_pred)
        predicciones.append({
            "date": fecha_pred.date(),
            "temp": model_temp.predict(X_scaled)[0],
            "precip": model_precip.predict(X_scaled)[0],
            "humidity": model_humidity.predict(X_scaled)[0],
            "uvindex": model_uv.predict(X_scaled)[0]
        })
    return pd.DataFrame(predicciones)

# Generar predicciones para la semana
df_pred = generar_predicciones(fecha_actual)

# Mostrar predicción del día seleccionado
pred_hoy = df_pred[df_pred["date"] == fecha_actual.date()].iloc[0]
st.markdown("### 📊 Predicciones del día seleccionado")
col1, col2 = st.columns(2)
with col1:
    st.metric("🌡️ Temperatura", f"{round(pred_hoy['temp'], 2)} °C")
    st.metric("💧 Humedad", f"{round(pred_hoy['humidity'], 2)} %")
with col2:
    st.metric("🌧️ Precipitación", f"{round(pred_hoy['precip'], 2)} mm")
    st.metric("🔆 Índice UV", f"{round(pred_hoy['uvindex'], 2)}")

st.caption("*📌 Nota: las predicciones se basan en registros diarios. Cada valor representa una estimación mayoritaria para ese día.*")

# Gráficas semanales con expander
with st.expander("### 📈 Evolución semanal de las variables"):
    tabs = st.tabs([ "🌧️ Precipitación", "💧 Humedad", "🔆 Índice UV"])

    with tabs[0]:
        fig_precip = px.bar(df_pred, x="date", y="precip", title="Precipitación diaria (mm)", labels={"precip": "Precipitación"})
        st.plotly_chart(fig_precip, use_container_width=True)

    with tabs[1]:
        fig_hum = px.line(df_pred, x="date", y="humidity", title="Humedad diaria (%)", labels={"humidity": "Humedad"})
        st.plotly_chart(fig_hum, use_container_width=True)

    with tabs[2]:
        uv_min = df_pred["uvindex"].min()
        uv_max = df_pred["uvindex"].max()
        margen = (uv_max - uv_min) * 0.2 if uv_max != uv_min else 0.1

        fig_uv = px.scatter(
            df_pred,
            x="date",
            y="uvindex",
            title="Índice UV diario",
            labels={"uvindex": "Índice UV"},
            size="uvindex",
            color="uvindex",
            color_continuous_scale="Blues"
        )
        fig_uv.update_yaxes(autorange="reversed", range=[uv_min - margen, uv_max + margen])
        st.plotly_chart(fig_uv, use_container_width=True)

# Comparador múltiple de fechas
st.markdown("### 🔍 Comparar predicciones entre fechas")
fechas_comparar = st.multiselect("Selecciona varias fechas futuras (máx 5):", sorted(fechas_validas), max_selections=5)
df_comparacion = pd.DataFrame()

for f in fechas_comparar:
    pred_df = generar_predicciones(pd.to_datetime(f))
    pred_df["seleccion"] = str(f)
    df_comparacion = pd.concat([df_comparacion, pred_df], ignore_index=True)

if not df_comparacion.empty:
    colx, coly = st.columns(2)
    with colx:
        fig_precip = px.line(df_comparacion, x="date", y="precip", color="seleccion", title="Comparativa de precipitación", labels={"precip": "Precipitación (mm)"})
        st.plotly_chart(fig_precip, use_container_width=True)
    with coly:
        fig_hum = px.line(df_comparacion, x="date", y="humidity", color="seleccion", title="Comparativa de humedad", labels={"humidity": "Humedad (%)"})
        st.plotly_chart(fig_hum, use_container_width=True)

# Calculadora de riesgo solar
st.markdown("### ☀️ Calculadora de riesgo solar")
uv = round(pred_hoy["uvindex"], 2)

if uv < 3:
    st.success(f"🟢 Riesgo bajo ({uv}). Puedes exponerte al sol con precaución.")
    st.caption("Usa gafas de sol en días brillantes. Considera protector solar si vas a estar al aire libre más de 1h.")
elif 3 <= uv < 6:
    st.warning(f"🟡 Riesgo moderado ({uv}). Usa protector solar y evita horas punta.")
    st.caption("Recomendado FPS 30+, gafas de sol y sombrero. Evita exponerte entre las 12 y las 16h.")
else:
    st.error(f"🔴 Riesgo alto ({uv}). Evita exposición prolongada entre 12 y 16h.")
    st.caption("Obligatorio protector solar FPS 50+, gafas, ropa clara y sombra. Revisa también apps de radiación solar local.")

# Gráfico UV semanal
st.markdown("#### 📉 Evolución del índice UV esta semana")
df_pred["Riesgo"] = pd.cut(df_pred["uvindex"], bins=[0, 3, 6, 11], labels=["Bajo", "Moderado", "Alto"], include_lowest=True)
colores_uv = {"Bajo": "green", "Moderado": "orange", "Alto": "red"}
uv_min = df_pred["uvindex"].min()
uv_max = df_pred["uvindex"].max()
margen = (uv_max - uv_min) * 0.5 if uv_max != uv_min else 0.2

fig_uv_riesgo = px.bar(
    df_pred,
    x="date",
    y="uvindex",
    color="Riesgo",
    color_discrete_map=colores_uv,
    title="Índice UV diario y nivel de riesgo",
    labels={"uvindex": "Índice UV", "date": "Fecha"}
)
fig_uv_riesgo.update_yaxes(range=[uv_min - margen, uv_max + margen])
st.plotly_chart(fig_uv_riesgo, use_container_width=True)

