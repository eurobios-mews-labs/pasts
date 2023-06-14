import pandas as pd

from darts.datasets import AirPassengersDataset
from darts.models import AutoARIMA, Prophet, ExponentialSmoothing
from darts.utils.utils import ModelMode, SeasonalityMode

from seriestemporelles.signal.signal_analysis import SignalAnalysis
from seriestemporelles.signal.visualization import Visualisation

if __name__ == '__main__':
    # ---- Load data ---
    series = AirPassengersDataset().load()
    dt = pd.DataFrame(series.values())
    dt.rename(columns={0: 'passengers'}, inplace=True)
    dt.index = series.time_index

    # ---- Vizualise data ---
    Visualisation(dt).plot_signal()
    Visualisation(dt).plot_smoothing()
    signal = SignalAnalysis(dt)
    signal.profiling()
    report = signal.apply_test('stationary', 'kpss')

    # ---- Machine Learning ---
    timestamp = '1957-06-01'
    signal.split_cv(timestamp=timestamp)

    signal.apply_model(AutoARIMA())
    signal.apply_model(Prophet())

    param_grid = {'trend': [ModelMode.ADDITIVE, ModelMode.MULTIPLICATIVE, ModelMode.NONE],
                  'seasonal': [SeasonalityMode.ADDITIVE, SeasonalityMode.MULTIPLICATIVE, SeasonalityMode.NONE],
                  }
    signal.apply_model(ExponentialSmoothing(), gridsearch=True, parameters=param_grid)

    # --- get some results ---
    print(signal.scores)
    exp_smoothing_pred = signal.results['ExponentialSmoothing']['predictions']
    exp_smoothing_params = signal.results['ExponentialSmoothing']['best_parameters']

    # ---  Vizualise the predictions ---
    signal.show_predictions()
