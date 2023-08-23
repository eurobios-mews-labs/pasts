import pandas as pd
from darts.datasets import AirPassengersDataset, AustralianTourismDataset
from darts.models import ExponentialSmoothing, AutoARIMA, Prophet, XGBModel, VARIMA
from darts.utils.utils import ModelMode, SeasonalityMode

from pasts.signal import DecomposedSignal
from pasts.visualization import Visualisation
from sklearn.linear_model import LinearRegression
import numpy as np

# Univariate
series = AirPassengersDataset().load()
dt = pd.DataFrame(series.values())
dt.rename(columns={0: 'passengers'}, inplace=True)
dt.index = series.time_index
# dt.loc[pd.to_datetime('1960-04-01')] = np.nan

lm = LinearRegression()
X = [i for i in range(0, len(dt))]
X = np.reshape(X, (len(X), 1))
y = dt.values
X_new = [i for i in range(0, len(dt) + 6)]
X_new = np.reshape(X_new, (len(X_new), 1))
lm.fit(X, y)
trend = lm.predict(X_new)




# ---- Visualize data ---
signal = DecomposedSignal(dt)
signal.detrend()
print(signal.properties)
Visualisation(signal).plot_signal()
Visualisation(signal).acf_plot()
signal.apply_stat_test('stationary')
signal.apply_stat_test('stationary', 'kpss')
signal.apply_stat_test('seasonality')

# ---- Machine Learning ---
timestamp = '1959-01-01'
signal.validation_split(timestamp=timestamp)

signal.apply_model(ExponentialSmoothing())

signal.apply_model(AutoARIMA())
signal.apply_model(Prophet())

param_grid = {'trend': [ModelMode.ADDITIVE, ModelMode.MULTIPLICATIVE, ModelMode.NONE],
              'seasonal': [SeasonalityMode.ADDITIVE, SeasonalityMode.MULTIPLICATIVE, SeasonalityMode.NONE],
              }
signal.apply_model(ExponentialSmoothing(), gridsearch=True, parameters=param_grid)

# --- Compute scores ---
signal.compute_scores()
signal.compute_scores(axis=0)

# ---  Visualize the predictions ---
Visualisation(signal).show_predictions()

# --- Aggregated Model ---
signal.apply_aggregated_model([ExponentialSmoothing(), Prophet()])
signal.compute_scores(axis=1)
signal.compute_scores(axis=0)
Visualisation(signal).show_predictions()

# --- Forecast ---
signal.forecast("Prophet", 15)
signal.forecast("AutoARIMA", 15)
signal.forecast("ExponentialSmoothing", 15)
signal.forecast("AggregatedModel", 15)
Visualisation(signal).show_forecast()

# Multivariate
series_m = AustralianTourismDataset().load()[['Hol', 'VFR', 'Oth']]
df_m = pd.DataFrame(series_m.values())
df_m.rename(columns={0: 'Hol', 1: 'VFR', 2: 'Oth'}, inplace=True)
df_m.index = pd.date_range(start='2020-01-01', periods=len(df_m), freq='MS')


X = [i for i in range(len(df_m))]
X = np.reshape(X, (len(X), 1))
df_trends = pd.DataFrame(index=pd.date_range(start=df_m.index[0], periods=len(df_m)+6, freq=df_m.index.freq))
X_new = [i for i in range(len(df_m) + 6)]
X_new = np.reshape(X_new, (len(X_new), 1))
for col in df_m.columns:
    y = df_m[col].values
    lm = LinearRegression()
    lm.fit(X, y)
    df_trends[col] = lm.predict(X_new)





signal_m = DecomposedSignal(df_m)
signal_m.detrend()
print(signal_m.properties)
signal_m.apply_stat_test('causality')
Visualisation(signal_m).plot_signal()

# ---- Machine Learning ---
timestamp = '2022-06-01'
signal_m.validation_split(timestamp=timestamp)

signal_m.apply_model(XGBModel(lags=[-1, -2, -3]))

param_grid = {'trend': [None, 'c'],
              'q': [0, 1, 2]}

signal_m.apply_model(VARIMA(), gridsearch=True, parameters=param_grid)

signal_m.compute_scores(axis=1)

# --- Aggregated Model ---
signal_m.apply_aggregated_model([XGBModel(lags=[-1, -2, -3]), VARIMA()])
signal_m.compute_scores(axis=0)

# ---  Visualize the predictions ---
Visualisation(signal_m).show_predictions()

# --- Forecast ---
signal_m.forecast("AggregatedModel", 50)
signal_m.forecast("VARIMA", 10)
signal_m.forecast("XGBModel", 10)
Visualisation(signal_m).show_forecast()
