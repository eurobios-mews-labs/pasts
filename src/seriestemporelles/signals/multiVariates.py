from seriestemporelles.signals.signal_abs import Signals
import pandas as pd


class MultiSignal(Signals):
    def __init__(self, data):
        self.data = data

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.report = super().profiling()
