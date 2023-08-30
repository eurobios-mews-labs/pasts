import warnings

import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot


class Visualization:
    """
    A class to visualize signals.

    Attributes
    ----------
    __signal : Signal

    Methods
    -------
    plot_signal():
        Plots raw data and transformed data if operations have been applied.
    plot_smoothing(resample_size, window_size):
        Plots resampled data.
    acf_plot():
        Plots autocorrelation (only for univariate series).
    show_predictions():
        Plots raw data and predicted values on same graph.
    show_forecast():
        Plots raw data and forecasted values (for future dates) on same graph.

    See also

    --------
    pasts.signal for details on the Signal object.
    pasts.operations for details on operations on time series.
    pasts.model for details on predictions and forecast methods.
    """

    def __init__(self, signal: "Signal"):
        """
        Constructs all the necessary attributes for the Visualization object.

        Parameters
        ----------
        signal : Signal
        """
        self.__signal = signal

    def plot_signal(self, **kwargs):
        """
        Plots raw data and transformed data if operations have been applied.
        """
        fig, ax = plt.subplots()
        i = 0
        self.__signal.data.plot(ax=ax, **kwargs)
        for col in self.__signal.data.columns:
            ax.lines[i].set_label(f'actual series {col}')
            i += 1
        if (self.__signal.data - self.__signal.rest_data).sum().sum() != 0:
            self.__signal.rest_data.plot(ax=ax, **kwargs)
            for col in self.__signal.rest_data.columns:
                ax.lines[i].set_label(f'transformed series {col}')
                i += 1
        plt.show()

    def plot_smoothing(self, resample_size: str = 'A', window_size: int = 12):
        """
         :param resample_size: calendar for resample, 'A' : year, 'D': Day, 'M': Month
         :type window_size: int : the size of window to rolling
         """
        resample_yr = self.__signal.data.resample(resample_size).mean()
        roll_yr = self.__signal.data.rolling(window_size).mean()
        ax = self.__signal.data.plot(alpha=0.5, style='-')  # store axis (ax) for latter plots
        resample_yr.plot(style=':', ax=ax)
        roll_yr.plot(style='--', ax=ax)
        ax.legend(['Resample at year frequency', 'Rolling average (smooth), window size=%s' % window_size])

    def acf_plot(self):
        """
        Plots autocorrelation (only for univariate series)
        """
        if not self.__signal.properties['is_univariate']:
            raise Exception('Can only plot acf for univariate series')
        autocorrelation_plot(self.__signal.data)

    def show_predictions(self):
        """
        Plots raw data and predicted values on same graph.
        """
        fig, ax = plt.subplots()
        if not self.__signal.models:
            raise Exception('No predictions have been computed.')
        n_signals = self.__signal.test_data.shape[1]

        labels = ['Actuals_s' + str(i) for i in range(1, n_signals + 1)]
        ax.plot(self.__signal.data, c='gray')
        for model in self.__signal.models.keys():
            pred = pd.DataFrame(self.__signal.models[model]['predictions'].values())
            pred.columns = self.__signal.models[model]['predictions'].columns
            pred.index = self.__signal.models[model]['predictions'].time_index
            ax.plot(pred)
            list_model_label = [model + '_s' + str(i) for i in range(1, n_signals + 1)]
            labels += list_model_label
        ax.legend(labels)
        plt.xlabel('time')
        plt.ylabel('values')
        plt.show()

    def show_forecast(self):
        """
        Plots raw data and forecasted values (for future dates) on same graph.
        """
        fig, ax = plt.subplots()
        n_signals = self.__signal.test_data.shape[1]

        labels = ['Actuals_s' + str(i) for i in range(1, n_signals + 1)]
        ax.plot(self.__signal.data, c='gray')
        last_obs = self.__signal.data.iloc[-1:]
        for model in self.__signal.models.keys():
            if 'forecast' not in self.__signal.models[model]:
                warnings.warn(f'No forecasts have been computed with {model}')
                continue
            pred = pd.DataFrame(self.__signal.models[model]['forecast'].values())
            pred.columns = self.__signal.models[model]['forecast'].columns
            pred.index = self.__signal.models[model]['forecast'].time_index
            pred = pd.concat([last_obs, pred])
            ax.plot(pred)
            list_model_label = [model + '_s' + str(i) for i in range(1, n_signals + 1)]
            labels += list_model_label
        ax.legend(labels)
        plt.xlabel('time')
        plt.ylabel('values')
        plt.show()
