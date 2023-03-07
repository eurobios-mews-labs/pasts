from abc import ABC
from typing import Dict, Any

import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

list_test_uni_variate = [adfuller, kpss]


class Signals(ABC):

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def profiling(self, head: int = 5) -> Dict[str, Any]:
        """

        :rtype:  Dict[str, Any]
        """
        profiling: Dict[str, Any] = {'shape': self.data.shape, 'types': self.data.dtypes, 'head': self.data.head(head),
                                     'nanSum': self.data.isnull().sum(),
                                     'quantiles': self.data.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T}

        return profiling

    def plot_signal(self):
        self.data.plot(figsize=(20, 10), linewidth=5, fontsize=20)

    def plot_smoothing(self, resample_size: str = 'A', window_size: int = 12):
        """
         :param resample_size: calendar for resample, 'A' : year, 'D': Day, 'M': Month
         :type window_size: int : the size of window to rolling
         """
        resample_yr = self.data.resample(resample_size).mean()
        roll_yr = self.data.rolling(window_size).mean()
        ax = self.data.plot(alpha=0.5, style='-')  # store axis (ax) for latter plots
        resample_yr.plot(style=':', ax=ax)
        roll_yr.plot(style='--', ax=ax)
        ax.legend(['Resample at year frequency', 'Rolling average (smooth), window size=%s' % window_size])
