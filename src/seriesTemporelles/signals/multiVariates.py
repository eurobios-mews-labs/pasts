from seriesTemporelles.signals.signal_abs import Signals


class MultiSignal(Signals):
    def __init__(self, data):
        self.data = data

    # def profiling(self, head=5):
    #     profiling = super(MultiSignal, self).profiling()
    #     return profiling
    #
    # def plot_signal(self):
    #     super(MultiSignal, self).plot_signal()
    #     print('plot of data signal')
    #
    # def plot_smoothing(self, resample_size: str = 'A', window_size: int = 12):
    #     super(MultiSignal, self).plot_smoothing()
    #
    #
