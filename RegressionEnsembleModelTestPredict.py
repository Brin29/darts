import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NaiveDrift, NaiveSeasonal, LinearRegressionModel, RegressionEnsembleModel
from darts.metrics import mape
from datetime import datetime

# Cargar datos
df = pd.read_csv('./content/ventas.csv')
serie = TimeSeries.from_dataframe(df, 'fecha', 'ventas')

# Modelos base
models = [
    NaiveDrift(),
    NaiveSeasonal(K=12)  # Usa 12 si es mensual
]

# Modelo de regresión para el ensemble
regression_model = LinearRegressionModel(
    quantiles=[0.25, 0.5, 0.75],
    lags_future_covariates=[0],
    likelihood='quantile',
    fit_intercept=False,
)

# Crear ensemble
ensemble_model = RegressionEnsembleModel(
    forecasting_models=models,
    regression_train_n_points=12,
    regression_model=regression_model,
)

# Inicializar
serie_actual = serie
forecast_total = None
anios_a_predecir = 10

# Predecir año por año
for i in range(anios_a_predecir):
    print(f"📅 Prediciendo el año {serie_actual.end_time().year + 1}...")

    # Entrenar con los datos actuales
    ensemble_model.fit(serie_actual)

    # Predecir el siguiente año (12 meses si es mensual)
    forecast = ensemble_model.predict(n=12)

    # Guardar CSV con la predicción de ese año
    year = serie_actual.end_time().year + 1
    forecast.to_dataframe().to_csv(f'./forecast_ano_{year}.csv')

    # Agregar el forecast a la serie actual
    serie_actual = serie_actual.append(forecast)

    # Acumular todas las predicciones
    forecast_total = forecast if forecast_total is None else forecast_total.append(forecast)

# Graficar resultados
serie.plot(label='Histórico')
forecast_total.plot(label='Predicción 10 años')
plt.title('Predicción año por año hasta 2035')
plt.legend()
plt.tight_layout()
plt.show()
