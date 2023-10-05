# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

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

    def plot_signal(self, **kwargs) -> None:
        """
        Plots raw data and transformed data if operations have been applied.
        """
        fig, ax = plt.subplots()
        legend = []
        i = 0
        self.__signal.data.plot(ax=ax, **kwargs)
        for col in self.__signal.data.columns:
            legend.append(f'raw data: {col}')
            i += 1
        if (self.__signal.data - self.__signal.rest_data).sum().sum() != 0:
            self.__signal.rest_data.plot(ax=ax, **kwargs)
            for col in self.__signal.rest_data.columns:
                legend.append(f'transformed data: {col}')
                i += 1
        plt.legend(legend)
        if self.__signal.operation_train is not None:
            if self.__signal.operation_train.dict_op:
                plt.title(f'Operations to transform data: {list(self.__signal.operation_data.dict_op.keys())}',
                          fontdict={'fontsize': 10})
        plt.show()

    def acf_plot(self) -> None:
        """
        Plots autocorrelation (only for univariate series)
        """
        if not self.__signal.properties['is_univariate']:
            raise Exception('Can only plot acf for univariate series')
        autocorrelation_plot(self.__signal.data)

    def show_predictions(self) -> None:
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

    def show_forecast(self) -> None:
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
