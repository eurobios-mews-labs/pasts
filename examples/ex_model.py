import pandas as pd

from darts.datasets import AirPassengersDataset, AustralianTourismDataset
from darts.models import AutoARIMA, Prophet, ExponentialSmoothing, XGBModel, VARIMA
from darts.utils.utils import ModelMode, SeasonalityMode

from pasts.signal import Signal
from pasts.visualization import Visualization


if __name__ == '__main__':

    # ----- Univariate -----

    # ---- Load data ----
    series = AirPassengersDataset().load()
    dt = pd.DataFrame(series.values())
    dt.rename(columns={0: 'passengers'}, inplace=True)
    dt.index = series.time_index

    # --- Visualize data ---
    signal = Signal(dt)
    print(signal.properties)
    Visualization(signal).plot_signal()
    Visualization(signal).acf_plot()
    signal.apply_stat_test('stationary')
    signal.apply_stat_test('stationary', 'kpss')
    signal.apply_stat_test('seasonality')

    # --- Machine Learning ---
    timestamp = '1958-12-01'
    signal.validation_split(timestamp=timestamp)
    signal.apply_operations(['trend', 'seasonality'])
    Visualization(signal).plot_signal()

    signal.apply_model(ExponentialSmoothing())

    signal.apply_model(AutoARIMA())
    signal.apply_model(Prophet())

    # If trend and seasonality have been removed, cannot perform this gridsearch
    param_grid = {'trend': [ModelMode.ADDITIVE, ModelMode.MULTIPLICATIVE, ModelMode.NONE],
                 'seasonal': [SeasonalityMode.ADDITIVE, SeasonalityMode.MULTIPLICATIVE, SeasonalityMode.NONE],
                 }
    signal.apply_model(ExponentialSmoothing(), gridsearch=True, parameters=param_grid)

    # --- Compute scores ---
    signal.compute_scores()
    signal.compute_scores(axis=0)

    # ---  Visualize predictions ---
    Visualization(signal).show_predictions()

    # --- Aggregated Model ---
    signal.apply_aggregated_model([AutoARIMA(), Prophet()])
    signal.compute_scores(axis=1)
    Visualization(signal).show_predictions()

    # --- Forecast ---
    signal.forecast("Prophet", 100)
    signal.forecast("AggregatedModel", 100)
    signal.forecast("AutoARIMA", 100)
    signal.forecast("ExponentialSmoothing", 100)

    # --- Visualize forecasts ---
    Visualization(signal).show_forecast()

    # ----- Multivariate -----

    # ---- Load data ----
    series_m = AustralianTourismDataset().load()[['Hol', 'VFR', 'Oth']]
    df_m = pd.DataFrame(series_m.values())
    df_m.rename(columns={0: 'Hol', 1: 'VFR', 2: 'Oth'}, inplace=True)
    df_m.index = pd.date_range(start='2020-01-01', freq='MS', periods=len(df_m)) # index must be a date

    # --- Visualize data ---
    signal_m = Signal(df_m)
    print(signal_m.properties)
    signal_m.apply_stat_test('causality')
    Visualization(signal_m).plot_signal()

    # --- Machine Learning ---
    timestamp = '2022-06-01'
    signal_m.validation_split(timestamp=timestamp)
    signal_m.apply_operations(['trend'])
    Visualization(signal_m).plot_signal()
    signal_m.apply_model(XGBModel(lags=[-1, -2, -3]))

    param_grid = {'trend': [None, 'c'],
                  'q': [0, 1, 2]}

    signal_m.apply_model(VARIMA(), gridsearch=True, parameters=param_grid)

    signal_m.compute_scores(axis=1)

    # --- Aggregated Model ---
    signal_m.apply_aggregated_model([XGBModel(lags=[-1, -2, -3]), VARIMA()])
    signal_m.compute_scores()

    # ---  Visualize predictions ---
    Visualization(signal_m).show_predictions()

    # --- Forecast ---
    signal_m.forecast("AggregatedModel", 50)
    signal_m.forecast("VARIMA", 50)
    signal_m.forecast("XGBModel", 50)

    # --- Visualize forecasts ---
    Visualization(signal_m).show_forecast()
