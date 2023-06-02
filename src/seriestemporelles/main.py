import pandas as pd
import matplotlib.pyplot as plt

from darts.datasets import AirPassengersDataset
from darts.models import AutoARIMA, Prophet, ExponentialSmoothing

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
    signal.apply_model(ExponentialSmoothing())

    # --- get some results ---
    print(signal.scores)
    exp_smoothing_pred = signal.results['ExponentialSmoothing']['predictions']

    # ---  Vizualise the predictions ---
    df_predictions = signal.test_set.copy()
    for model in signal.results.keys():
        df_predictions.loc[:, model] = signal.results[model]['predictions']

    plt.plot(signal.data, c='gray')
    plt.plot(df_predictions['AutoARIMA'], c='blue')
    plt.plot(df_predictions['Prophet'], c='green')
    plt.plot(df_predictions['ExponentialSmoothing'], c='red')

    cols = df_predictions.columns  # .drop('passengers')
    plt.legend(cols)
