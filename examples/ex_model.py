import pandas as pd

from darts.datasets import AirPassengersDataset, AustralianTourismDataset
from darts.models import AutoARIMA, ExponentialSmoothing, XGBModel, VARIMA, Prophet, RandomForest
from darts.utils.utils import ModelMode, SeasonalityMode

from pasts.signal import Signal
from pasts.visualization import Visualization


if __name__ == '__main__':

    # ----- Univariate -----

    # --- Load data ---
    series = AirPassengersDataset().load()
    dt = pd.DataFrame(series.values())
    dt.rename(columns={0: 'passengers'}, inplace=True)
    dt.index = series.time_index

    # --- Create a Signal object ---
    # The path argument indicates in which directory to save fitted models
    signal = Signal(dt, path='examples/AirPassenger1')

    # --- Analyze data ---
    print(signal.properties)
    Visualization(signal).plot_signal()
    Visualization(signal).acf_plot()
    signal.apply_stat_test('stationary')
    signal.apply_stat_test('stationary', 'kpss')
    signal.apply_stat_test('seasonality')

    # ---- Machine Learning ----
    # --- Split data between train and test ---
    timestamp = '1958-12-01'
    signal.validation_split(timestamp=timestamp)

    # --- Remove trend and seasonality ---
    signal.apply_operations(['trend', 'seasonality'])
    Visualization(signal).plot_signal()

    # --- Apply models ---
    # save_model=True indicates that the fitted estimator and its predictions will be saved
    # in a joblib file in signal.path
    signal.apply_model(ExponentialSmoothing(), save_model=True)
    signal.apply_model(AutoARIMA(), save_model=True)
    signal.apply_model(Prophet(), save_model=True)
    signal.apply_model(RandomForest(lags=24), save_model=True)

    # If trend and seasonality have been removed, you cannot perform this gridsearch !
    param_grid = {'trend': [ModelMode.ADDITIVE, ModelMode.MULTIPLICATIVE, ModelMode.NONE],
                  'seasonal': [SeasonalityMode.ADDITIVE, SeasonalityMode.MULTIPLICATIVE, SeasonalityMode.NONE],
                  }
    signal.apply_model(ExponentialSmoothing(), gridsearch=True, parameters=param_grid)

    # --- Compute scores ---
    signal.compute_scores()  # unit-wise by default
    signal.compute_scores(axis=0)  # time-wise

    # ---  Visualize predictions ---
    Visualization(signal).show_predictions()

    # --- Aggregated Model ---
    signal.apply_aggregated_model([ExponentialSmoothing(), Prophet(), RandomForest(lags=24)], save_model=True)
    signal.compute_scores(axis=1)

    # --- Confidence intervals ---
    signal.compute_conf_intervals(window_size=3)

    # Plot only the aggregate predictions and the associated interval
    Visualization(signal).show_predictions(aggregated_only=True)
    Visualization(signal).show_predictions_plotly()

    # --- Forecast ---
    # Generate forecasts for 100 future dates and save fitted estimator and predictions in joblib files.
    # If you want to generate more forecasts later, simply use signal.get_saved_models and the models will not need
    # to be fitted again.
    signal.forecast("AggregatedModel", 100, save_model=True)
    signal.forecast("AutoARIMA", 100, save_model=True)
    signal.forecast("ExponentialSmoothing", 100, save_model=True)
    signal.forecast("Prophet", 100, save_model=True)
    signal.forecast("RandomForest", 100, save_model=True)

    # --- Confidence intervals ---
    signal.compute_conf_intervals(window_size=3)

    # --- Visualize forecasts ---
    Visualization(signal).show_forecast()
    Visualization(signal).show_forecast_plotly()

    # ----- Multivariate -----

    # --- Load data ---
    series_m = AustralianTourismDataset().load()[['Hol', 'VFR', 'Oth']]
    df_m = pd.DataFrame(series_m.values())
    df_m.rename(columns={0: 'Hol', 1: 'VFR', 2: 'Oth'}, inplace=True)
    df_m.index = pd.date_range(start='2020-01-01', freq='MS', periods=len(df_m)) # index must be a date

    # --- Create a Signal object ---
    signal_m = Signal(df_m, path='examples/AustralianTourism1')

    # --- Analyze data ---
    print(signal_m.properties)
    signal_m.apply_stat_test('causality')
    Visualization(signal_m).plot_signal()

    # ---- Machine Learning ----
    # --- Split data between train and test ---
    timestamp = '2022-06-01'
    signal_m.validation_split(timestamp=timestamp)

    # --- Remove trend ---
    signal_m.apply_operations(['trend'])
    Visualization(signal_m).plot_signal()

    # --- Apply models ---
    signal_m.apply_model(XGBModel(lags=[-1, -2, -3]), save_model=True)

    param_grid = {'trend': [None, 'c'],
                  'q': [0, 1, 2]}

    signal_m.apply_model(VARIMA(), gridsearch=True, parameters=param_grid)

    signal_m.compute_scores(axis=1)  # time-wise

    # --- Aggregated Model ---
    signal_m.apply_aggregated_model([XGBModel(lags=[-1, -2, -3]), VARIMA()], save_model=True)
    signal_m.compute_scores()  # unit-wise

    # ---  Visualize predictions ---
    Visualization(signal_m).show_predictions(aggregated_only=True)

    # --- Forecast ---
    signal_m.forecast("AggregatedModel", 50, save_model=True)
    signal_m.forecast("VARIMA", 50, save_model=True)
    signal_m.forecast("XGBModel", 50, save_model=True)

    # --- Visualize forecasts ---
    Visualization(signal_m).show_forecast()
