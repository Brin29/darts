import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import ExponentialSmoothing

df = pd.read_csv('./content/ventas.csv')
serie = TimeSeries.from_dataframe(df, 'fecha', 'ventas')

serie_tr, serie_ts = serie.split_before(0.7)

model_es = ExponentialSmoothing()
model_es.fit(serie_tr)
probabilistic_forecast = model_es.predict(len(36), num_samples=500)

serie.plot(label='Actual')
probabilistic_forecast.plot(label='Probali')