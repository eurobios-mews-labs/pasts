from seriesTemporelles.signals.signal_abs import Signals
from seriesTemporelles.test.test_statistiques import TestStatistics


class UniSignal(Signals, TestStatistics):
    def __init__(self, data):
        self.data = data

    def profiling(self, head=5):
        self.profiling = super(UniSignal, self).profiling()
        return self.profiling

    def plot_signal(self):
        super(UniSignal, self).plot_signal()
        print('plot of data signal')

    def plot_smoothing(self, resample_size: str = 'A', window_size: int = 12):
        super(UniSignal, self).plot_smoothing()

    def is_stationary(self, test_stat, *args, **kwargs):
        test_output = super(UniSignal, self).is_stationary(test_stat, *args, **kwargs)
        self.profiling[test_stat.__name__] = test_output
        return self.profiling


