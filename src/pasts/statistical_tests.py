from itertools import combinations

import pandas as pd

from darts import TimeSeries
from darts.utils.statistics import check_seasonality
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests

dict_test_uni_variate = {'stationary': ['kpss', 'adfuller'],
                         'seasonality': ['check_seasonality']
                         }

dict_test_multi_variate = {'causality': ['grangercausalitytests']}

dict_test = {'stationary': ['kpss', 'adfuller'],
             'seasonality': 'check_seasonality',
             'causality': 'grangercausalitytests'}


def check_arguments(
        condition: bool,
        message: str = ""):
    """
    Checks  boolean condition and raises a ValueError.
    """

    if condition:
        raise TypeError(message)


class TestStatistics:
    """
    A class to apply statistical tests on a Signal object.

    Attributes
    ----------
    __signal : Signal
        Passed Signal object

    Methods
    -------
    is_stationary(alpha, test_name, *args, **kwargs) :
        Applies a stationarity test on entire dataset.
    is_seasonal(*args, **kwargs) :
        Tests seasonality in signal.
    test_causality(alpha, maxlag, *args, **kwargs) :
        Tests granger causality between features.
    apply(type_test, test_stat_name, *args, **kwargs) :
        Applies requested test.
    """

    def __init__(self, signal: "Signal"):
        """
        Constructs all the necessary attributes for the TestStatistics object.

        Parameters
        ----------
        signal : Signal
                Signal object on which to apply tests.
        """
        self.__signal = signal

    def is_stationary(self, alpha: int = 0.05, test_name='kpss', *args, **kwargs):
        """
        Applies a stationarity test on the entire dataset.

        Parameters
        ----------
        test_name : str
            Name of stationarity test to use
        alpha : float
            Significance level (default=0.05)

        Returns
        -------
        Whether the null hypothesis is rejected and p-value
        """

        if test_name == 'adfuller':
            """
               Results of Dickey-Fuller Test, with :\n'
               'Null Hypothesis (HO): The time series is non-stationary. In other words,'
               ' it has some time-dependent structure and does not have constant variance over time.\n'
               'Alternate Hypothesis(HA): Series is stationary .'
            alpha: 
            """
            df_test = adfuller(self.__signal.data, *args, **kwargs)
            df_output = pd.Series(df_test[0:4],
                                  index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
            return df_output['p-value'] <= alpha, df_output['p-value']

        else:
            """
              'Results of KPSS Test, with :\n'
               'Null Hypothesis (HO): Series is trend stationary or series has no unit root.\n'
               'Alternate Hypothesis(HA): Series is non-stationary or series has a unit root.
            """
            kpss_test = kpss(self.__signal.data, *args, **kwargs)
            kpss_output = pd.Series(kpss_test[0:3], index=['Test Statistic', 'p-value', '#Lags Used'])
            return kpss_output['p-value'] > alpha, kpss_output['p-value']

    def is_seasonal(self, *args, **kwargs):
        """
        Tests seasonality in signal.

        Returns
        -------
        Whether a seasonality was detected and the seasonality period.
        """
        return check_seasonality, check_seasonality(TimeSeries.from_dataframe(
            self.__signal.data), *args, **kwargs)

    def test_causality(self, alpha: int = 0.05, maxlag: int = 1, *args, **kwargs):
        """
        Tests granger causality between features.

        Parameters
        ----------
        alpha : float
            Significance level (default=0.05)
        maxlag : int
            Computes the test for all lags up to maxlag

        Returns
        -------
        Dictionary containing tests results.
        keys: 'feature1 --> feature2'
        values: Whether the null hypothesis (absence of correlation) is rejected and the p-value.
        """
        names = self.__signal.data.columns
        results = {}
        pairs = list(combinations(names, 2))
        for pair in pairs:
            series1, series2 = pair
            result_1_2 = grangercausalitytests(self.__signal.data[[series1, series2]], maxlag, verbose=False,
                                               *args, **kwargs)
            result_2_1 = grangercausalitytests(self.__signal.data[[series2, series1]], maxlag, verbose=False,
                                               *args, **kwargs)
            pvalue1 = result_1_2[1][0]['ssr_ftest'][1]
            pvalue2 = result_2_1[1][0]['ssr_ftest'][1]
            results[f"{series1}-->{series2}"] = (pvalue1 <= alpha, pvalue1)
            results[f"{series2}-->{series1}"] = (pvalue2 <= alpha, pvalue2)
        return results

    def apply(self, type_test: str, test_stat_name: str, *args, **kwargs):
        """
        Applies requested test.

        Parameters
        ----------
        type_test : str
            Type of test to apply ('stationary', 'seasonality' or 'causality')
        test_stat_name : str
            Name of test to apply

        Returns
        -------
        Result of corresponding test function
        """
        if self.__signal.properties['is_univariate']:
            dict_test_ = dict_test_uni_variate
        else:
            dict_test_ = dict_test_multi_variate

        check_arguments(
            type_test not in dict_test_.keys(),
            f"Select the type of statistical test from: {dict_test_.keys()}.",
        )

        check_arguments(
            test_stat_name not in dict_test_[type_test],
            f"Select correct test from: {dict_test_[type_test]}.",
        )

        if type_test == 'stationary':
            return self.is_stationary(test_name=test_stat_name, *args, **kwargs)
        elif type_test == 'seasonality':
            return self.is_seasonal(*args, **kwargs)
        else:
            return self.test_causality(*args, **kwargs)
