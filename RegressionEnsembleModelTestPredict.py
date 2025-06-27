import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NaiveDrift, NaiveSeasonal, LinearRegressionModel, RegressionEnsembleModel
from darts.metrics import mape

# Cargar datos
df = pd.read_csv('./content/ventas.csv')
serie = TimeSeries.from_dataframe(df, 'fecha', 'ventas')

# Crear modelos base
models = [
    NaiveDrift(),               # Modelo de tendencia
    NaiveSeasonal(K=12)         # Modelo de estacionalidad (mensual: 12 meses)
]

# Definir modelo de regresión para combinar ambos
quantiles = [0.25, 0.5, 0.75]
regression_model = LinearRegressionModel(
    quantiles=quantiles,
    lags_future_covariates=[0],
    likelihood='quantile',
    fit_intercept=False,
)

# Crear el modelo ensemble
ensemble_model = RegressionEnsembleModel(
    forecasting_models=models,
    regression_train_n_points=12,
    regression_model=regression_model,
)

# Entrenar con toda la serie
ensemble_model.fit(serie)

# Calcular cuántos pasos faltan hasta 2025
from datetime import datetime

fecha_objetivo = pd.Timestamp("2025-12-01")
if serie.freq == "M":
    steps = (fecha_objetivo.year - serie.end_time().year) * 24 + (fecha_objetivo.month - serie.end_time().month)
elif serie.freq == "W":
    steps = ((fecha_objetivo - serie.end_time()).days) // 7
elif serie.freq == "D":
    steps = (fecha_objetivo - serie.end_time()).days
else:
    raise ValueError("Frecuencia no soportada")

print(f"Pasos hasta 2025: {steps}")

# Realizar la predicción
forecast = ensemble_model.predict(n=120)

# Graficar resultados
serie.plot(label='Datos históricos')
forecast.plot(label='Predicción hasta 2025')
plt.title("Predicción de ventas con tendencia + estacionalidad")
plt.legend()
plt.show()
