from abc import ABC, abstractmethod
from typing import *

import pandas as pd
from darts import TimeSeries
from darts.utils.statistics import check_seasonality
from statsmodels.tsa.stattools import adfuller, kpss

dict_test_uni_variate = {'stationary': ['kpss', 'adfuller'],
                         'seasonality': ['check_seasonality']
                         }

dict_test_multi_variate = {'causality': ['grangercausalitytests']}


class Signal(ABC):
    def __init__(self, data: pd.DataFrame):
        self.__data = data

    @property
    def data(self):
        return self.__data

    #@abstractmethod
    def split_cv(self, timestamp, n_splits_cv=None):
        pass

    #@abstractmethod
    def apply_model(self):
        pass

    #@abstractmethod
    def show_predictions(self):
        pass

    @property
    def n_columns(self):
        """Number of components (dimensions) contained in the series."""
        return self.data.shape[1]

    @property
    def is_uni_variate(self):
        """Whether this series is uni_variate."""
        return self.n_columns == 1

    def profiling(self, head: int = 5) -> Dict[str, Any]:
        """

        :rtype:  Dict[str, Any]
        """
        profiling: Dict[str, Any] = {'shape': self.data.shape, 'types': self.data.dtypes, 'head': self.data.head(head),
                                     'nanSum': self.data.isnull().sum(),
                                     'quantiles': self.data.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T}

        return profiling

    def is_stationary(self, alpha: int = 0.05, test_name='kpss', *args, **kwargs):
        """

        :param test_name: nae of test statistic used to test stationary
        :param alpha: significance level
        """

        if test_name == 'adfuller':
            """
               Results of Dickey-Fuller Test, with :\n'
               'Null Hypothesis (HO): The time series is non-stationary. In other words,'
               ' it has some time-dependent structure and does not have constant variance over time.\n'
               'Alternate Hypothesis(HA): Series is stationary .'
            alpha: 
            """
            # eval('test_name' + "()")
            df_test = adfuller(self.data, *args, **kwargs)
            df_output = pd.Series(df_test[0:4],
                                  index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
            if df_output['p-value'] > alpha:
                return False, df_output['p-value']
            else:
                return True, df_output['p-value']

        else:
            """
              'Results of KPSS Test, with :\n'
               'Null Hypothesis (HO): Series is trend stationary or series has no unit root.\n'
               'Alternate Hypothesis(HA): Series is non-stationary or series has a unit root.
            """
            kpss_test = kpss(self.data, *args, **kwargs)
            kpss_output = pd.Series(kpss_test[0:3], index=['Test Statistic', 'p-value', '#Lags Used'])
            if kpss_output['p-value'] > alpha:
                return test_name, True, kpss_output['p-value']
            else:
                return test_name, False, kpss_output['p-value']

    def is_seasonality(self, *args, **kwargs):
        return check_seasonality, check_seasonality(self.series, *args, **kwargs)
