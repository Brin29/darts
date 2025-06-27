import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NaiveDrift
from darts.models import NaiveSeasonal

df = pd.read_csv('./content/ventas.csv')
serie = TimeSeries.from_dataframe(df, 'fecha', 'ventas')

serie_tr, serie_ts = serie.split_before(0.7)
# serie_tr.plot(label='Entrenamiento')
# serie_ts.plot(label='Prueba')

seasonal_model = NaiveSeasonal(K=12)
seasonal_model.fit(serie_tr)

seasonal_forecast = seasonal_model.predict(64)

print(serie_tr)

drift_model = NaiveDrift()
drift_model.fit(serie_tr)
drift_forecast = drift_model.predict(64)

combined_forecast = drift_forecast + seasonal_forecast - serie_tr.last_value()

serie.plot()
combined_forecast.plot(label='combined')
drift_forecast.plot(label='drift')
# print(serie.duration)
# print(serie.freq)
# print(serie.n_timesteps)


plt.show()