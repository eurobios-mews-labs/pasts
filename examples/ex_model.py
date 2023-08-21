import pandas as pd

from darts.datasets import AirPassengersDataset, AustralianTourismDataset
from darts.models import AutoARIMA, Prophet, ExponentialSmoothing, XGBModel, VARIMA
from darts.utils.utils import ModelMode, SeasonalityMode

from pasts import Signal
from pasts import Visualisation


if __name__ == '__main__':
    # Univariate
    # ---- Load data ---
    series = AirPassengersDataset().load()
    dt = pd.DataFrame(series.values())
    dt.rename(columns={0: 'passengers'}, inplace=True)
    dt.index = series.time_index
    # dt.loc[pd.to_datetime('1960-04-01')] = np.nan

    # ---- Visualize data ---
    signal = Signal(dt)
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

    # Multivariate
    series_m = AustralianTourismDataset().load()[['Hol', 'VFR', 'Oth']]
    df_m = pd.DataFrame(series_m.values())
    df_m.rename(columns={0: 'Hol', 1: 'VFR', 2: 'Oth'}, inplace=True)
    df_m.index = series_m.time_index

    signal_m = Signal(df_m)
    print(signal_m.properties)
    signal_m.apply_stat_test('causality')
    Visualisation(signal_m).plot_signal()

    # ---- Machine Learning ---
    timestamp = 30
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


