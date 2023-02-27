import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
###############################################
from pandas.plotting import autocorrelation_plot
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.tsa.stattools import kpss
from sys import exit
from sklearn.model_selection import TimeSeriesSplit
from darts.metrics import mape


class Signal:

    def __init__(self, data: pd.DataFrame) -> None:

        """

        :type data: data contained a times series in columns and date time in index
        """
        self.times_series = data
        if type(data.index) == pd.core.indexes.datetimes.DatetimeIndex:
            return

        print("Error in type of index, should be datetimes")
        exit()
        self.time_step = (self.times_series.index[1] - self.times_series.index[0])

    def profiling(self, head=5):
        print("##################### Shape #####################")
        print(self.times_series.shape)
        print("##################### Types #####################")
        print(self.times_series.dtypes)
        print("##################### Head #####################")
        print(self.times_series.head(head))
        print("##################### Tail #####################")
        print(self.times_series.tail(head))
        print("##################### NA #####################")
        print(self.times_series.isnull().sum())
        print("##################### Quantiles #####################")
        print(self.times_series.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

    def plot_signal(self):
        self.times_series.plot(figsize=(20, 10), linewidth=5, fontsize=20)
        # plt.xlabel('Year', fontsize=20)

    def plot_smoothing(self, resample_size: str = 'A', window_size: int = 12):
        """
         :param resample_size: calendar for resample, 'A' : year, 'D': Day, 'M': Month
         :type window_size: int : the size of window to rolling
         """
        resample_yr = self.times_series.resample(resample_size).mean()
        roll_yr = self.times_series.rolling(window_size).mean()
        ax = self.times_series.plot(alpha=0.5, style='-')  # store axis (ax) for latter plots
        resample_yr.plot(style=':', ax=ax)
        roll_yr.plot(style='--', ax=ax)
        ax.legend(['Resample at year frequency', 'Rolling average (smooth), window size=%s' % window_size])

    def adf_test(self, alpha: object = 0.05, auto_lag: object = 'AIC') -> None:
        """

         :param alpha: significance level
         :type auto_lag: object : {"AIC", "BIC", "t-stat", None}
         Method to use when automatically determining the lag length among the
         values 0, 1, ..., maxlag.
         """
        print('Results of Dickey-Fuller Test, with :\n'
              'Null Hypothesis (HO): The time series is non-stationary.\n'
              'In other words, it has some time-dependent structure and does not have constant variance over time.\n'
              'Alternate Hypothesis(HA): Series is stationary .')

        dftest = adfuller(self.times_series, autolag=auto_lag)
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)
        if dfoutput['p-value'] <= alpha:
            print('The time series is stationary')
        else:
            print('The p-value is obtained is greater than significance level of %s. '
                  'So, the time series is non-stationary' % alpha)

    def kpss_test(self, alpha: object = 0.05) -> None:
        """

         :param alpha: significance level
         """
        print('Results of KPSS Test, with :\n'
              'Null Hypothesis (HO): Series is trend stationary or series has no unit root.\n'
              'Alternate Hypothesis(HA): Series is non-stationary or series has a unit root.')
        kpss_test = kpss(self.times_series, regression='c', nlags="auto")
        kpss_output = pd.Series(kpss_test[0:3], index=['Test Statistic', 'p-value', '#Lags Used'])
        for key, value in kpss_test[3].items():
            kpss_output['Critical Value (%s)' % key] = value
        print(kpss_output)
        if kpss_output['p-value'] <= alpha:
            print('The time series is non-stationary')
        else:
            print('The p-value is obtained is greater than significance level of %s. '
                  'So, the time series is trend stationary' % alpha)

    def split_cv(self, n_splits):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return tscv.split(TimeSeries.from_dataframe(self.times_series))

    def apply_model(self, model, train_index=[], test_index=[],
                    parameters: list = {}, gridsearch: bool = False):
        data_train = self.times_series.iloc[train_index]
        data_test = self.times_series.iloc[test_index]
        if gridsearch:
            best_model, best_parametres = model.gridsearch(parameters=parameters,
                                                           series=data_train,
                                                           start=0.5,
                                                           forecast_horizon=12)
        best_model = model
        best_model.fit(data_train)
        forecast = model.predict(len(data_test))
        print('model {} obtains MAPE: {:.2f}%'.format(model, mape(data_test, forecast)))
