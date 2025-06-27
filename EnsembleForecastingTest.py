import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NaiveEnsembleModel, NaiveDrift, NaiveSeasonal
from darts.metrics import mape

df = pd.read_csv('./content/ventas.csv')
serie = TimeSeries.from_dataframe(df, 'fecha', 'ventas')

serie_tr, serie_ts = serie.split_before(0.7)

models = [NaiveDrift(), NaiveSeasonal(12)]

ensemble_model = NaiveEnsembleModel(forecasting_models=models)

backset = ensemble_model.historical_forecasts(
  serie_tr, start=0.6, forecast_horizon=3, verbose=True
)

print('MAPE = %.2f' % (mape(backset, serie_tr)))
serie_tr.plot()
backset.plot()

plt.show()