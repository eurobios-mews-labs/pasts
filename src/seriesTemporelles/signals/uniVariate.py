from abc import ABC

import pandas as pd
from typing import Dict, Any, Union, Tuple

from seriesTemporelles.signals.signal_abs import Signals
from seriesTemporelles.test.test_statistiques import TestStatistics


class UniSignal(Signals, TestStatistics):

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.report = super().profiling()

    def is_stationary(self, test_stat, *args, **kwargs):
        test_output = super(UniSignal, self)._is_stationary(test_stat, *args, **kwargs)
        self.report[test_stat.__name__] = test_output
        return self.report