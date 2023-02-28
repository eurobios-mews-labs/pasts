from darts.datasets import AirPassengersDataset
import pandas as pd
from seriesTemporelles.signals.uniSignal import UniSignal
from statsmodels.tsa.stattools import kpss



if __name__ == '__main__':
    series = AirPassengersDataset().load()
    dt = pd.DataFrame(series.values())
    dt.rename(columns={0: 'passengers'})
    dt.index = series.time_index
    signal = UniSignal(dt)
    signal.profiling()
    signal.plot_signal()
    signal.plot_smoothing()
    report = signal.is_stationary(test_stat=kpss, regression='c', nlags="auto")

