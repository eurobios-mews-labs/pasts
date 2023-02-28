import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.tsa.stattools import kpss
import matplotlib.pyplot as plt
from sys import exit
from pandas.plotting import autocorrelation_plot



class Exploratory_Data_Analysis:
    def __init__(self, data: pd.DataFrame) -> None:
        """

        :type data: data contained a times series in columns and date time in index
        """
        self.data = data
        if not type(data.index) == pd.core.indexes.datetimes.DatetimeIndex:
            print("Error in type of index, should be datetimes")
            exit()

    def check_df(self, head=5):
        print("##################### Shape #####################")
        print(self.data.shape)
        print("##################### Types #####################")
        print(self.data.dtypes)
        print("##################### Head #####################")
        print(self.data.head(head))
        print("##################### Tail #####################")
        print(self.data.tail(head))
        print("##################### NA #####################")
        print(self.data.isnull().sum())
        print("##################### Quantiles #####################")
        print(self.data.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

    def plot_st(self):
        self.data.plot(figsize=(20, 10), linewidth=5, fontsize=20)
        plt.xlabel('Year', fontsize=20)

    def plot_smoothing(self, times_series_var: str, resample_size: str = 'A',
                       window_size: int = 12):
        """
        :param resample_size: calendar for resample, 'A' : year, 'D': Day, 'M': Month
        :type times_series_var: the name of times series in the data
        :type window_size: int : the size of window to rolling
        """
        times_series = self.data[times_series_var]
        resample_yr = times_series.resample(resample_size).mean()
        roll_yr = times_series.rolling(window_size).mean()
        ax = times_series.plot(alpha=0.5, style='-')  # store axis (ax) for latter plots
        resample_yr.plot(style=':', label='Resample at year frequency', ax=ax)
        roll_yr.plot(style='--', label='Rolling average (smooth), window size=%s' % window_size, ax=ax)
        ax.legend()


class Test_times_series_stationary:
    def __init__(self, data: pd.Series):
        self.time_series = data

    def adf_test(self, alpha: object = 0.05, auto_lag: object = 'AIC') -> None:
        """

        :param alpha: significance level
        :type auto_lag: object : {"AIC", "BIC", "t-stat", None}
        Method to use when automatically determining the lag length among the
        values 0, 1, ..., maxlag.
        """
        print('Results of Dickey-Fuller Test, with :\n'
              'Null Hypothesis (HO): The time series is non-stationary. In other words,'
              ' it has some time-dependent structure and does not have constant variance over time.\n'
              'Alternate Hypothesis(HA): Series is stationary .')

        dftest = adfuller(self.time_series, autolag=auto_lag)
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)
        if dfoutput['p-value'] > alpha:
            print('The p-value is obtained is greater than significance level of %s. '
                  'So, the time series is non-stationary' % alpha)
        else:
            print('The time series is stationary')

    def kpss_test(self, alpha: object = 0.05) -> None:
        """

        :param alpha: significance level
        """
        print('Results of KPSS Test, with :\n'
              'Null Hypothesis (HO): Series is trend stationary or series has no unit root.\n'
              'Alternate Hypothesis(HA): Series is non-stationary or series has a unit root.')
        kpss_test = kpss(self.time_series, regression='c', nlags="auto")
        kpss_output = pd.Series(kpss_test[0:3], index=['Test Statistic', 'p-value', '#Lags Used'])
        for key, value in kpss_test[3].items():
            kpss_output['Critical Value (%s)' % key] = value
        print(kpss_output)
        if kpss_output['p-value'] > alpha:
            print('The p-value is obtained is greater than significance level of %s. '
                  'So, the time series is trend stationary' % alpha)
        else:
            print('The time series  is non-stationary')


class Operation_on_ts:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def dtrend_times_series(self, list_times_series: list):
        data_avg = pd.concat([self.data[var].rolling(12).mean() for var in list_times_series], axis=1)
        data_dtrend = self.data[list_times_series] - data_avg
        return data_dtrend

    def acf(self, times_series):
        autocorrelation_plot(times_series)

    # def differencing(self):
    #
    # def seasonal_adjustment(self):

# class Evaluation_Metrics:
#     def __init__(self):
#
#     def mean_absolute_error(self):
#
#     def mean_squared_eError(self):
#
#     def root_mean_squared_error(self):
#
#
#
#
