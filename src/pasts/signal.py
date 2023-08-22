import copy
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from darts import TimeSeries
from statsmodels.tsa.seasonal import seasonal_decompose

from pasts.model import Model, AggregatedModel
from pasts.test_statistiques import TestStatistics, dict_test
from pasts.validation import Validation
from pasts.metrics import Metrics

from sklearn.linear_model import LinearRegression
from scipy.fftpack import fft, ifft


def profiling(data: pd.DataFrame):
    """
    Finds some properties about time series.

    Parameters
    ----------
    data: pd.Dataframe
        Dataframe of time series with time as index and entities as columns

    Returns
    -------
    Dictionary of properties
    """
    return {'shape': data.shape,
            'types': data.dtypes,
            'is_univariate': data.shape[1] == 1,
            'nanSum': data.isnull().sum(),
            'quantiles': data.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T}


class SignalAbstract(ABC):
    """
    A class to represent a signal.

    Attributes
    ----------
    properties : dict
        Info about the series (shape, types, univariate or multivariate, number of missing values, quantiles)
    tests_stat : dict
        keys: names of statistical tests applied on the series
        values: Whether the null hypothesis is rejected, p-value
    models : dict
        keys: models applied on the series
        values: dictionary of predictions and best parameters
    performance_models: dict
        keys: names of metrics
        values: dataframe of metrics computed for each model

    Methods
    -------
    apply_stat_test(type_test, test_stat_name = None, *args, **kwargs):
        Applies statistical test to the univariate or multivariate series.
    validation_split(timestamp, n_splits_cv = None):
        Splits the series between train and test sets.
    apply_model(model, gridsearch = False, parameters = None):
        Applies statistical, machine learning of deep learning model to the series.
    compute_scores(list_metrics: list[str] = None, axis=1)
        Computes scores of models on test data.
    apply_aggregated_model(list_models, refit=False)
        Aggregates a given list of models according to their performance on test data.
    forecast(model_name: str, horizon: int)
        Generates forecasts for future dates.
        """

    def __init__(self, data: pd.DataFrame):
        """
        Constructs all the necessary attributes for the signal object.

        Parameters
        ----------
        data : pd.Dataframe
                Dataframe of time series with time as index and one or several entities as columns.
        """
        self.__data = data
        self.__properties = profiling(data)
        self.__tests_stat = {}
        self.__train_data = None
        self.__test_data = None
        self.__cv_tseries = None
        self.models = {}
        self.__performance_models = {}

    @property
    def data(self):
        return self.__data

    @property
    def train_data(self):
        return self.__train_data

    @property
    def test_data(self):
        return self.__test_data

    @property
    def properties(self):
        return self.__properties

    @property
    def tests_stat(self):
        return self.__tests_stat

    @property
    def performance_models(self):
        return self.__performance_models

    def apply_stat_test(self, type_test: str, test_stat_name: str = None, *args, **kwargs):
        """
        Applies statistical test to the univariate or multivariate series.

        Fills the attribute tests_stat.

        Parameters
        ----------
        type_test : str
                Type of test to be applied: stationary or seasonality for univariate series, causality for multivariate
        test_stat_name : str, optional
                adfuller or kpss if type_test is stationary (default is adfuller)
                ignored if type_test is seasonality or causality

        Returns
        -------
        None
        """
        call_test = TestStatistics(self)
        if (type_test == 'stationary') & (test_stat_name is None):
            test_stat_name = 'adfuller'
            self.tests_stat[f"{type_test}: {test_stat_name}"] = call_test.apply(type_test, test_stat_name,
                                                                                *args, **kwargs)
        elif (type_test == 'stationary') & (test_stat_name is not None):
            self.tests_stat[f"{type_test}: {test_stat_name}"] = call_test.apply(type_test, test_stat_name,
                                                                                *args, **kwargs)
        else:
            test_stat_name = dict_test[type_test]
            self.tests_stat[f"{type_test}: {test_stat_name}"] = call_test.apply(type_test, test_stat_name,
                                                                                *args, **kwargs)

    def validation_split(self, timestamp, n_splits_cv=None):
        """
        Splits the series between train and test sets.

        If n_splits_cv is filled, yields train and test indices for cross-validation.

        Fills the attributes train_data and test_data.

        Parameters
        ----------
        timestamp :
                Time index to split between train and test sets
        n_splits_cv : int, optional
                Number of folds for cross-validation

        Returns
        -------
        None
        """
        call_validation = Validation(self.data)
        call_validation.split_cv(timestamp, n_splits_cv)
        self.__train_data = call_validation.train_data
        self.__test_data = call_validation.test_data
        self.__cv_tseries = call_validation.cv_tseries

    def compute_scores(self, list_metrics: list[str] = None, axis=1):
        """
        Computes scores of models on test data.

        Fills the attribute models with the scores and the attribute performance_models.

        Parameters
        ----------
        list_metrics : list[str]
                List of name of metrics chosen in ['r2', 'mse', 'rmse', 'mape', 'smape', 'mae']
        axis : int, optional (default = 1)
                Whether to compute scores unit-wise (axis=1) or time-wise (axis=0)

        Returns
        -------
        None
        """
        if axis == 1:
            score_type = 'unit_wise'
        elif axis == 0:
            score_type = 'time_wise'
        else:
            raise ValueError('axis must be 0 or 1')
        if list_metrics is None:
            list_metrics = ['r2', 'mse', 'rmse', 'mape', 'smape', 'mae']
        call_metric = Metrics(self, list_metrics)
        for model in self.models.keys():
            self.models[model]['scores'][score_type] = call_metric.compute_scores(model, axis)
        self.performance_models[score_type] = call_metric.scores_comparison(axis)

    def apply_aggregated_model(self, list_models, refit=False):
        """
        Aggregates a given list of models according to their performance on test data.

        Creates a new model "AggregatedModel" in the attribute models.

        Parameters
        ----------
        list_models :
                List of instances of models.
        refit : bool, optional
                Whether to refit estimators even if they were previously fitted (default is False).
                Ignored for estimators not previously fitted.

        Returns
        -------
        None
        """
        dict_models = {model.__class__.__name__: model for model in list_models}
        if refit:
            for model in dict_models.values():
                self.apply_model(model)
        else:
            for model_name, model in dict_models.items():
                if model_name not in self.models.keys():
                    warnings.warn(f'{model_name}  has not yet been fitted. Fitting {model_name}...', UserWarning)
                    self.apply_model(model)
        self.models['AggregatedModel'] = AggregatedModel(self).apply(dict_models)

    @abstractmethod
    def apply_model(self, model, gridsearch=False, parameters=None):
        """
        Applies statistical, machine learning of deep learning model to the series.

        Fills the attribute models.

        Parameters
        ----------
        model
                Instance of a model. Will be refitted even if is has already been.
        gridsearch : bool, optional
                Whether to perform a gridsearch (default is False)
        parameters : pd.Dataframe, optional
                Parameters to test if a gridsearch is performed (default is None)

        Returns
        -------
        None
        """

    @abstractmethod
    def forecast(self, model_name: str, horizon: int):
        """
        Generates forecasts for future dates.

        Fills models attribute with a 'forecast' key and a 'final estimator' key.

        Parameters
        ----------
        model_name : str
                Name of a model. If AggregatedModel, forecasts will be computed with the models included in the aggregation.
        horizon : int
                Horizon of prediction.

        Returns
        -------
        None
        """


class Signal(SignalAbstract):
    """
    A class to represent a signal.

    Attributes
    ----------
    properties : dict
        Info about the series (shape, types, univariate or multivariate, number of missing values, quantiles)
    tests_stat : dict
        keys: names of statistical tests applied on the series
        values: Whether the null hypothesis is rejected, p-value
    models : dict
        keys: models applied on the series
        values: dictionary of predictions and best parameters
    performance_models: dict
        keys: names of metrics
        values: dataframe of metrics computed for each model

    Methods
    -------
    apply_stat_test(type_test, test_stat_name = None, *args, **kwargs):
        Applies statistical test to the univariate or multivariate series.
    validation_split(timestamp, n_splits_cv = None):
        Splits the series between train and test sets.
    apply_model(model, gridsearch = False, parameters = None):
        Applies statistical, machine learning of deep learning model to the series.
    compute_scores(list_metrics: list[str] = None, axis=1)
        Computes scores of models on test data.
    apply_aggregated_model(list_models, refit=False)
        Aggregates a given list of models according to their performance on test data.
    forecast(model_name: str, horizon: int)
        Generates forecasts for future dates.
        """

    def __init__(self, data: pd.DataFrame):
        """
        Constructs all the necessary attributes for the signal object.

        Parameters
        ----------
        data : pd.Dataframe
                Dataframe of time series with time as index and one or several entities as columns.
        """
        super().__init__(data)

    def apply_model(self, model, gridsearch=False, parameters=None):
        """
        Applies statistical, machine learning of deep learning model to the series.

        Fills the attribute models.

        Parameters
        ----------
        model
                Instance of a model. Will be refitted even if is has already been.
        gridsearch : bool, optional
                Whether to perform a gridsearch (default is False)
        parameters : pd.Dataframe, optional
                Parameters to test if a gridsearch is performed (default is None)

        Returns
        -------
        None
        """
        self.models[model.__class__.__name__] = Model(self).apply(copy.deepcopy(model), gridsearch, parameters)

    def forecast(self, model_name: str, horizon: int):
        """
        Generates forecasts for future dates.

        Fills models attribute with a 'forecast' key and a 'final estimator' key.

        Parameters
        ----------
        model_name : str
                Name of a model. If AggregatedModel, forecasts will be computed with the models included in the aggregation.
        horizon : int
                Horizon of prediction.

        Returns
        -------
        None
        """
        if model_name == 'AggregatedModel':
            if 'AggregatedModel' not in self.models.keys():
                raise Exception('Aggregated Model has not been trained. Use method apply_aggregated_model first.')
            for model in self.models['AggregatedModel']['models'].keys():
                self.models[model]['final_estimator'] = Model(self).compute_final_estimator(model)
                self.models[model]['forecast'] = self.models[model]['final_estimator'].predict(horizon)
            self.models['AggregatedModel']['forecast'] = AggregatedModel(self).compute_final_estimator()
        else:
            if model_name not in self.models.keys():
                raise Exception(f'{model_name} has not been trained.')
            self.models[model_name]['final_estimator'] = Model(self).compute_final_estimator(model_name)
            self.models[model_name]['forecast'] = self.models[model_name]['final_estimator'].predict(horizon)


class DecomposedSignal(SignalAbstract):

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.__detrend_signals = pd.DataFrame(np.zeros((len(self.data.index), len(self.data.columns))),
                                              columns=self.data.columns, index=self.data.index)
        self.__trend_linear = pd.DataFrame(np.zeros((len(self.data.index), len(self.data.columns))),
                                           columns=self.data.columns, index=self.data.index)
        self.__trend_periodic = pd.DataFrame(np.zeros((len(self.data.index), len(self.data.columns))),
                                             columns=self.data.columns, index=self.data.index)
        self.__data_raw = pd.DataFrame(np.zeros((len(self.data.index), len(self.data.columns))),
                                       columns=self.data.columns, index=self.data.index)
        self.__operations = []

    @property
    def operations(self):
        return self.__operations

    @property
    def trend_linear(self):
        return self.__trend_linear

    @trend_linear.setter
    def trend_linear(self, value):
        self.__trend_linear = value

    @property
    def trend_periodic(self):
        return self.__trend_periodic

    @trend_periodic.setter
    def trend_periodic(self, value):
        self.__trend_periodic = value

    @property
    def data_raw(self):
        return self.__data_raw

    def detrend(self):
        for i in self.data.columns:
            trend_linear = seasonal_decompose(self.data[i], model='additive').trend
            for j in np.arange(-6, 0):
                trend_linear[j] = (2 * trend_linear[j - 1] - trend_linear[j - 2])
                trend_linear[-j - 1] = (2 * trend_linear[-j] - trend_linear[-j + 1])
            self.trend_periodic[i] = seasonal_decompose(self.data[i], model='additive').seasonal
            self.trend_linear[i] = trend_linear
            self.data_raw[i] = list(self.data[i])
            self.data[i] = self.data[i] - self.trend_linear[i] - self.trend_periodic[i]
        self.operations.append('trend')

    def retrend_uni(self, fc):
        type_output = type(fc).__name__
        if 'trend' in self.operations:
            if type_output == 'TimeSeries':
                fc = fc.pd_dataframe()
            else:
                fc = fc.copy()
            for k in fc.columns:
                period = self.trend_periodic[0:12].copy()
                period.index = [t.strftime("%m") for t in period.index]
                step = self.trend_linear.iloc[-1] - self.trend_linear.iloc[-2]
                for t, val in enumerate(fc[k].copy()):
                    if fc.index[t] in self.trend_linear.index:
                        fc[k][t] = val + self.trend_linear.loc[fc.index[t]][self.trend_linear.columns[0]].copy() + self.trend_periodic.loc[
                            fc.index[t]][self.trend_periodic.columns[0]].copy()
                    else:
                        ID = fc[k].index[t].strftime("%m")
                        fc[k][t] = fc[k][t] + t * step + period.loc[ID]
            if type_output == "TimeSeries":
                fc = TimeSeries.from_dataframe(fc)
        return fc

    def retrend_multi(self, fc):
        type_output = type(fc).__name__
        if 'trend' in self.operations:
            if type_output == 'TimeSeries':
                fc = fc.pd_dataframe()
            else:
                fc = fc.copy()
            for k in fc.columns:
                period = self.trend_periodic[k][0:12]
                period.index = [t.strftime("%m") for t in period.index]
                step = self.trend_linear[k].iloc[-1] - self.trend_linear[k].iloc[-2]
                for t, val in enumerate(fc[k]):
                    if fc.index[t] in self.trend_linear[k].index:
                        fc[k][t] = val + self.trend_linear[k].loc[fc.index[t]].copy() + self.trend_periodic[k].loc[
                            fc.index[t]].copy()
                    else:
                        ID = fc[k].index[t].strftime("%m")
                        fc[k][t] = fc[k][t] + t * step + period.loc[ID]
            if type_output == "TimeSeries":
                fc = TimeSeries.from_dataframe(fc)
        return fc

    def extrapolate_trends(self, steps):
        new_index = pd.date_range(start=self.data.index[0],
                                  periods=len(self.data) + steps, freq=self.data.index.freq)
        nan_data = np.empty((steps, self.data.shape[1]))
        nan_data[:] = np.nan
        nan_df = pd.DataFrame(nan_data, index=new_index[-steps:], columns=self.data.columns)
        new_linear = pd.concat([self.trend_linear, nan_df])
        self.trend_linear = new_linear
        new_periodic = pd.concat([self.trend_periodic, nan_df])
        self.trend_periodic = new_periodic

        for i in self.data.columns:
            trend_values = self.trend_linear[i][:-steps]
            trend_length = len(trend_values)
            X = np.arange(trend_length).reshape(-1, 1)
            y = trend_values.to_numpy().reshape(-1, 1)

            model = LinearRegression()
            model.fit(X, y)

            future_steps = np.arange(trend_length, trend_length + steps).reshape(-1, 1)
            future_trend = model.predict(future_steps)
            self.trend_linear[i] = np.concatenate((trend_values, future_trend.flatten()))

            periodic_values = self.trend_periodic[i][:-steps]
            period = len(periodic_values)
            freqs = np.fft.fftfreq(period)
            fft_values = np.fft.fft(periodic_values)

            extrapolated_fft = np.concatenate((fft_values, fft_values[-1:] + (freqs[-1] + 1) * (
                    fft_values[-1] - fft_values[-2])))
            extrapolated_periodic = np.fft.ifft(extrapolated_fft).real
            self.trend_periodic[i] = np.concatenate((periodic_values, np.tile(
                extrapolated_periodic, steps // period + 1)[:steps]))

    def apply_model(self, model, gridsearch=False, parameters=None):
        model_name = model.__class__.__name__
        self.models[model_name] = Model(self).apply(copy.deepcopy(model), gridsearch, parameters)

        if 'trend' in self.operations:
            if self.properties['is_univariate']:
                self.models[model_name]['predictions'] = self.retrend_uni(self.models[model_name]['predictions'])
            else:
                self.models[model_name]['predictions'] = self.retrend_multi(self.models[model_name]['predictions'])

    def forecast(self, model_name: str, horizon: int):
        if model_name == 'AggregatedModel':
            if 'AggregatedModel' not in self.models.keys():
                raise Exception('Aggregated Model has not been trained. Use method apply_aggregated_model first.')
            for model in self.models['AggregatedModel']['models'].keys():
                self.models[model]['final_estimator'] = Model(self).compute_final_estimator(model)
                self.models[model]['forecast'] = self.models[model]['final_estimator'].predict(horizon)
                if 'trend' in self.operations:
                    if len(self.trend_linear) == len(self.data):
                        self.extrapolate_trends(horizon)
                    if self.properties['is_univariate']:
                        self.models[model]['forecast'] = self.retrend_uni(self.models[model]['forecast'])
                    else:
                        self.models[model]['forecast'] = self.retrend_multi(self.models[model]['forecast'])
            self.models['AggregatedModel']['forecast'] = AggregatedModel(self).compute_final_estimator()
        else:
            if model_name not in self.models.keys():
                raise Exception(f'{model_name} has not been trained.')
            self.models[model_name]['final_estimator'] = Model(self).compute_final_estimator(model_name)
            self.models[model_name]['forecast'] = self.models[model_name]['final_estimator'].predict(horizon)
            if 'trend' in self.operations:
                if len(self.trend_linear) == len(self.data):
                    self.extrapolate_trends(horizon)
                if self.properties['is_univariate']:
                    self.models[model_name]['forecast'] = self.retrend_uni(self.models[model_name]['forecast'])
                else:
                    self.models[model_name]['forecast'] = self.retrend_multi(self.models[model_name]['forecast'])
