import pandas as pd
from darts.datasets import AirPassengersDataset
from darts.models import ExponentialSmoothing, AutoARIMA, Prophet
from darts.utils.utils import ModelMode, SeasonalityMode

from pasts.operations import DecomposedSignal
from pasts.signal import Signal
from pasts.visualization import Visualisation

series = AirPassengersDataset().load()
dt = pd.DataFrame(series.values())
dt.rename(columns={0: 'passengers'}, inplace=True)
dt.index = series.time_index
# dt.loc[pd.to_datetime('1960-04-01')] = np.nan

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
Visualisation(signal).show_predictions()

# --- Forecast ---
signal.forecast("Prophet", 6)
signal.forecast("AggregatedModel", 6)
signal.forecast("AutoARIMA", 6)
signal.forecast("ExponentialSmoothing", 6)
Visualisation(signal).show_forecast()