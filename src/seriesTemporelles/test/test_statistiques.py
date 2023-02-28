from abc import ABC, abstractmethod

import pandas as pd


class TestStatistics(ABC):

    def __init__(self, data: pd.DataFrame):
        self.data = data

    @abstractmethod
    def is_stationary(self, test_stat, *args, **kwargs) -> object:
        test = test_stat(self.data, *args, **kwargs)
        test_output = pd.Series(test[0:3], index=['Test Statistic', 'p-value', '#Lags Used'])
        return test_output




