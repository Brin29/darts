import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries

df = pd.read_csv('./content/ventas.csv')
serie = TimeSeries.from_dataframe(df, 'fecha', 'ventas')

# print(serie.duration)
# print(serie.freq)
# print(serie.n_timesteps)
serie_tr, serie_ts = serie.split_before(0.7)
serie_tr.plot(label='Entrenamiento')
serie_ts.plot(label='Prueba')

plt.show()