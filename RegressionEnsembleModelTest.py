import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NaiveDrift, NaiveSeasonal, LinearRegressionModel, RegressionEnsembleModel
from darts.metrics import mape

df = pd.read_csv('./content/ventas.csv')
serie = TimeSeries.from_dataframe(df, 'fecha', 'ventas')

serie_tr, serie_ts = serie.split_before(0.7)

quantiles = [0.25, 0.5, 0.75]

models = [NaiveDrift(), NaiveSeasonal(12)]

regression_model = LinearRegressionModel(
  quantiles=quantiles,
  lags_future_covariates=[0],
  likelihood='quantile',
  fit_intercept=False,
)

ensemble_model = RegressionEnsembleModel(
  forecasting_models=models,
  regression_train_n_points=12,
  regression_model=regression_model,
)

backtest = ensemble_model.historical_forecasts(
  serie_tr, start=0.6, forecast_horizon=3, num_samples=500, verbose=True
)

print('MAPE = %.2f' % (mape(backtest, serie_tr)))
serie_tr.plot()
backtest.plot()
plt.show()