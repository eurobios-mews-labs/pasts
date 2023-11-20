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
import plotly.graph_objects as go
import matplotlib
import random

colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]


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

    def plot_signal(self, display=True, **kwargs) -> None:
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

        if display is True:
            plt.show()
        else:
            plt.close()

    def acf_plot(self) -> None:
        """
        Plots autocorrelation (only for univariate series)
        """
        if not self.__signal.properties['is_univariate']:
            raise Exception('Can only plot acf for univariate series')
        autocorrelation_plot(self.__signal.data)

    def show_predictions(self, aggregated_only=False, display=True) -> None:
        """
        Plots raw data and predicted values on same graph.
        """
        fig, ax = plt.subplots()
        if not self.__signal.models:
            raise Exception('No predictions have been computed.')
        n_signals = self.__signal.test_data.shape[1]

        labels = ['Actuals_s' + str(i) for i in range(1, n_signals + 1)]
        ax.plot(self.__signal.data, c='gray')
        if aggregated_only:
            if 'AggregatedModel' not in self.__signal.models.keys():
                raise Exception('No predictions have been computed with aggregated model')
            else:
                to_plot = ['AggregatedModel']
                for i, unit in enumerate(self.__signal.data.columns):
                    itv = self.__signal.models['AggregatedModel']['test_confidence_interval']
                    bounds = pd.DataFrame(index=self.__signal.models['AggregatedModel']['test_confidence_interval'].index)
                    bounds['lower'] = [interval[0] for interval in itv[unit].values]
                    bounds['upper'] = [interval[1] for interval in itv[unit].values]
                    ax.plot(bounds, color='green', linestyle='--')
                    ax.fill_between(bounds.index, bounds['lower'], bounds['upper'], color='green', alpha=0.3)
                    labels += [f'lower_s{str(i + 1)}', f'upper_s{str(i + 1)}', f'interval_s{str(i + 1)}']
        else:
            to_plot = self.__signal.models.keys()

        for model in to_plot:
            pred = pd.DataFrame(self.__signal.models[model]['predictions'].values())
            pred.columns = self.__signal.models[model]['predictions'].columns
            pred.index = self.__signal.models[model]['predictions'].time_index
            ax.plot(pred)
            list_model_label = [model + '_s' + str(i) for i in range(1, n_signals + 1)]
            labels += list_model_label

        ax.legend(labels)
        plt.xlabel('time')
        plt.ylabel('values')
        if display is True:
            plt.show()
        else:
            plt.close()

    def show_predictions_plotly(self):
        """
        Plots raw data and predicted values on the same Plotly graph with confidence intervals.
        """
        if not self.__signal.models:
            raise Exception('No predictions have been computed.')

        fig = go.Figure()

        # Plot actuals
        for i, unit in enumerate(self.__signal.data.columns):
            fig.add_trace(go.Scatter(x=self.__signal.data.index, y=self.__signal.data[unit], mode='lines',
                                     name=f'Actual_s{i + 1}', line=dict(color='#7f7f7f')))

        # Plot predictions and confidence intervals
        j = 0
        for model_name, model_data in self.__signal.models.items():

            pred = model_data['predictions']
            time_index = pred.time_index
            pred = pred.pd_dataframe()

            # Plot predictions
            for i, unit in enumerate(pred.columns):
                if j < len(colors):
                    trace_color = colors[j]
                else:
                    colors_ = dict(matplotlib.colors.cnames.items())
                    hex_colors = tuple(colors_.values())
                    trace_color = random.choice(hex_colors)

                trace_pred = go.Scatter(x=time_index, y=pred[unit], mode='lines', name=f'{model_name}_s{i + 1}',
                                        line=dict(color=trace_color))
                fig.add_trace(trace_pred)

                # Plot confidence interval
                itv = model_data['test_confidence_interval'][unit]
                bounds = pd.DataFrame(index=time_index)
                bounds['lower'] = [interval[0] for interval in itv.values]
                bounds['upper'] = [interval[1] for interval in itv.values]

                fig.add_trace(go.Scatter(x=time_index, y=bounds['lower'], mode='lines',
                                         line=dict(color=trace_color, dash='dash'), showlegend=False,
                                         legendgroup=f'CI_{model_name}_s{i+1}', name=f'CI_{model_name}_s{i+1}'))
                fig.add_trace(go.Scatter(x=time_index, y=bounds['upper'], mode='lines',
                                         line=dict(color=trace_color, dash='dash'), fill='tonexty',
                                         fillcolor=f'rgba{tuple(int(trace_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}',
                                         legendgroup=f'CI_{model_name}_s{i+1}', name=f'CI_{model_name}_s{i+1}'))
                j += 1

        # Configure layout
        fig.update_layout(
            title='Predictions with Confidence Intervals',
            xaxis_title='Time',
            yaxis_title='Values',
            showlegend=True
        )

        # Show the figure
        fig.show()

    def show_forecast(self, aggregated_only=False, display=True) -> None:
        """
        Plots raw data and forecasted values (for future dates) on same graph.
        """
        fig, ax = plt.subplots()
        n_signals = self.__signal.test_data.shape[1]

        labels = ['Actuals_s' + str(i) for i in range(1, n_signals + 1)]
        ax.plot(self.__signal.data, c='gray')
        last_obs = self.__signal.data.iloc[-1:]
        if aggregated_only:
            if 'AggregatedModel' not in self.__signal.models.keys():
                raise Exception('No predictions have been computed with aggregated model')
            else:
                to_plot = ['AggregatedModel']
                for i, unit in enumerate(self.__signal.data.columns):
                    itv = self.__signal.models['AggregatedModel']['forecast_confidence_interval']
                    bounds = pd.DataFrame(index=self.__signal.models['AggregatedModel'][
                        'forecast_confidence_interval'].index)
                    bounds['lower'] = [interval[0] for interval in itv[unit].values]
                    bounds['upper'] = [interval[1] for interval in itv[unit].values]
                    ax.plot(bounds, color='green', linestyle='--')
                    ax.fill_between(bounds.index, bounds['lower'], bounds['upper'], color='green', alpha=0.3)
                    labels += [f'lower_s{str(i + 1)}', f'upper_s{str(i + 1)}', f'interval_s{str(i + 1)}']
        else:
            to_plot = self.__signal.models.keys()

        for model in to_plot:
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
        if display is True:
            plt.show()
        else:
            plt.close()

    def show_forecast_plotly(self):
        """
        Plots raw data and predicted future values on the same Plotly graph with confidence intervals.
        """
        if not self.__signal.models:
            raise Exception('No predictions have been computed.')

        fig = go.Figure()

        # Plot actuals
        for i, unit in enumerate(self.__signal.data.columns):
            fig.add_trace(go.Scatter(x=self.__signal.data.index, y=self.__signal.data[unit], mode='lines',
                                     name=f'Actual_s{i + 1}', line=dict(color='#7f7f7f')))
        last_obs = self.__signal.data.iloc[-1:]

        # Plot predictions and confidence intervals
        j = 0
        for model_name, model_data in self.__signal.models.items():

            if 'forecast' not in self.__signal.models[model_name]:
                warnings.warn(f'No forecasts have been computed with {model_name}')
                continue

            pred = pd.DataFrame(model_data['forecast'].values())
            pred.columns = model_data['forecast'].columns
            pred.index = model_data['forecast'].time_index
            pred = pd.concat([last_obs, pred])

            # Plot predictions
            for i, unit in enumerate(pred.columns):
                if j < len(colors):
                    trace_color = colors[j]
                else:
                    colors_ = dict(matplotlib.colors.cnames.items())
                    hex_colors = tuple(colors_.values())
                    trace_color = random.choice(hex_colors)

                trace_pred = go.Scatter(x=pred.index, y=pred[unit], mode='lines', name=f'{model_name}_s{i + 1}',
                                        line=dict(color=trace_color))
                fig.add_trace(trace_pred)

                # Plot confidence interval
                itv = model_data['forecast_confidence_interval'][unit]
                bounds = pd.DataFrame(index=pred.index[1:])
                bounds['lower'] = [interval[0] for interval in itv.values]
                bounds['upper'] = [interval[1] for interval in itv.values]

                fig.add_trace(go.Scatter(x=itv.index, y=bounds['lower'], mode='lines',
                                         line=dict(color=trace_color, dash='dash'), showlegend=False,
                                         legendgroup=f'CI_{model_name}_s{i+1}', name=f'CI_{model_name}_s{i+1}'))
                fig.add_trace(go.Scatter(x=itv.index, y=bounds['upper'], mode='lines',
                                         line=dict(color=trace_color, dash='dash'), fill='tonexty',
                                         fillcolor=f'rgba{tuple(int(trace_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}',
                                         legendgroup=f'CI_{model_name}_s{i+1}', name=f'CI_{model_name}_s{i+1}'))
                j += 1

        # Configure layout
        fig.update_layout(
            title='Forecasts with Confidence Intervals',
            xaxis_title='Time',
            yaxis_title='Values',
            showlegend=True
        )

        # Show the figure
        fig.show()
