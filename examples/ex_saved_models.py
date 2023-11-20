import pandas as pd

from darts.datasets import AirPassengersDataset
# make sure to import all models even if not explicitly called in this code
# from darts.models import AutoARIMA, ExponentialSmoothing, XGBModel, VARIMA

from pasts.signal import Signal


if __name__ == '__main__':

    # ----- Univariate -----

    # ---- Load data ----
    series = AirPassengersDataset().load()
    dt = pd.DataFrame(series.values())
    dt.rename(columns={0: 'passengers'}, inplace=True)
    dt.index = series.time_index

    signal = Signal(dt, path='examples/AirPassenger1')

    # --- Get saved models ---
    signal.get_saved_models()
    signal.compute_conf_intervals(window_size=6)

    # --- Forecasts with models previously fitted only on train set ---
    signal.forecast("AggregatedModel", 100, save_model=True)
    signal.forecast("AutoARIMA", 100, save_model=True)
    signal.forecast("ExponentialSmoothing", 100, save_model=True)

    # --- Forecasts with models previously fitted only on entire dataset ---
    signal.forecast("AggregatedModel", 100)
    signal.forecast("AutoARIMA", 100)
    signal.forecast("ExponentialSmoothing", 100)