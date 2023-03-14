import pandas as pd

from seriestemporelles.signal.proporties_signal import Proprieties
from seriestemporelles.test.test_statistiques import TestStatistics


class SignalAnalysis(Proprieties):

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.report = super().profiling()

    def apply_test(self, type_test: str, test_stat_name: str, *args, **kwargs):
        call_test = TestStatistics(self.data)
        test_output = call_test.statistical_test(type_test, test_stat_name, *args, **kwargs)
        self.report[type_test] = test_output
        return self.report
