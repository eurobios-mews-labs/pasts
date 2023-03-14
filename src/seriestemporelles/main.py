import pandas as pd
from darts.datasets import AirPassengersDataset

from seriestemporelles.signal.signal_analysis import SignalAnalysis
from seriestemporelles.signal.visualization import Visualisation

if __name__ == '__main__':
    series = AirPassengersDataset().load()
    dt = pd.DataFrame(series.values())
    dt.rename(columns={0: 'passengers'})
    dt.index = series.time_index
    Visualisation(dt).plot_signal()
    Visualisation(dt).plot_smoothing()
    signal = SignalAnalysis(dt)
    signal.profiling()
    report = signal.apply_test('stationary', 'kpss')

