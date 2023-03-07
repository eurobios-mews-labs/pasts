from seriestemporelles.signal.signal_abstract import Signals
import pandas as pd


class Multi_variate(Signals):
    def __init__(self, data):
        self.data = data

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.report = super().profiling()
