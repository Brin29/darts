import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NaiveSeasonal

df = pd.read_csv('./content/ventas.csv')
serie = TimeSeries.from_dataframe(df, 'fecha', 'ventas')

serie_tr, serie_ts = serie.split_before(0.7)
serie_tr.plot(label='Entrenamiento')
serie_ts.plot(label='Prueba')

print(serie_tr)

naive_model = NaiveSeasonal(K=14)
naive_model.fit(serie_tr)
naive_forecast = naive_model.predict(124)

naive_forecast.plot(label="Naive forecast (K=1)")

# print(serie.duration)
# print(serie.freq)
# print(serie.n_timesteps)


plt.show()