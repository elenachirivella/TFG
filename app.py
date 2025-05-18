import streamlit as st
import pandas as pd
import joblib
from datetime import date, timedelta
import numpy as np

# Cargar scaler
scaler = joblib.load("scaler.pkl")

# Cargar cada modelo por separado
model_temp = joblib.load("rf_temp_futuro_30.pkl")
model_precip = joblib.load("rf_precip_futuro_30.pkl")
model_humidity = joblib.load("rf_humidity_futuro_30.pkl")
model_uv = joblib.load("rf_uvindex_futuro_30.pkl")

# Cargar dataset preprocesado
df_model = pd.read_csv("df_model.csv")
df_model["datetime"] = pd.to_datetime(df_model["datetime"])

# Título y descripción
st.title("Predicción Meteorológica - Valencia")
st.write("Esta aplicación predice condiciones climáticas futuras en Valencia: temperatura, precipitación, humedad e índice UV.")

# Selector de fecha futura
fecha_prediccion = st.date_input("Selecciona la fecha que quieres predecir:", date.today() + timedelta(days=1),
                                  min_value=df_model["datetime"].min().date() + timedelta(days=7),
                                  max_value=df_model["datetime"].max().date())

# Buscar los datos anteriores necesarios
fecha_actual = pd.to_datetime(fecha_prediccion)

# Buscar las fechas anteriores para los lags
lags = [1, 2, 3, 7]
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

# Extraer variables de fecha
inputs["day"] = fecha_actual.day
inputs["month"] = fecha_actual.month
inputs["year"] = fecha_actual.year
inputs["weekday"] = fecha_actual.weekday()
inputs["is_weekend"] = int(inputs["weekday"] in [5, 6])

# Crear DataFrame con una fila
X_pred = pd.DataFrame([inputs])

# Escalar
X_scaled = scaler.transform(X_pred)

# Predecir con cada modelo
pred_temp = model_temp.predict(X_scaled)[0]
pred_precip = model_precip.predict(X_scaled)[0]
pred_humidity = model_humidity.predict(X_scaled)[0]
pred_uv = model_uv.predict(X_scaled)[0]

# Unir resultados
predicciones = {
    'temp_futuro_30': pred_temp,
    'precip_futuro_30': pred_precip,
    'humidity_futuro_30': pred_humidity,
    'uvindex_futuro_30': pred_uv
}

# Mostrar resultados
st.subheader(f"Predicción para el {fecha_prediccion.strftime('%d/%m/%Y')}")
st.metric("Temperatura (°C)", round(predicciones['temp_futuro_30'], 2))
st.metric("Humedad (%)", round(predicciones['humidity_futuro_30'], 2))
st.metric("Precipitación (mm)", round(predicciones['precip_futuro_30'], 2))
st.metric("Índice UV", round(predicciones['uvindex_futuro_30'], 2))

# Evaluar riesgo solar
st.subheader("Calculadora de riesgo solar")
uv = predicciones['uvindex_futuro_30']
if uv < 3:
    st.success(f"Riesgo bajo ({uv:.1f}). Puedes exponerte al sol con precaución.")
elif 3 <= uv < 6:
    st.warning(f"Riesgo moderado ({uv:.1f}). Usa protector solar y evita horas punta.")
else:
    st.error(f"Riesgo alto ({uv:.1f}). Evita exposición prolongada entre 12 y 16h.")
