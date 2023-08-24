import copy
import warnings
from abc import ABC
from typing import Union

import pandas as pd
from darts import TimeSeries
from scipy import stats

from pasts.model import Model, AggregatedModel
from pasts.operations import Operation
from pasts.test_statistiques import TestStatistics, dict_test
from pasts.validation import Validation
from pasts.metrics import Metrics


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


class Signal(ABC):
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
        self.__rest_data = data.copy()
        self.operation_train = Operation()
        self.operation_data = Operation()
        self.__properties = profiling(data)
        self.__tests_stat = {}
        self.__train_data = None
        self.__test_data = None
        self.__rest_train_data = None
        self.__cv_tseries = None
        self.models = {}
        self.__performance_models = {}

    @property
    def data(self):
        return self.__data

    @property
    def rest_data(self):
        return self.__rest_data

    @property
    def train_data(self):
        return self.__train_data

    @property
    def test_data(self):
        return self.__test_data

    @property
    def rest_train_data(self):
        return self.__rest_train_data

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
        self.__rest_train_data = self.train_data.copy()

    def filter_outliers(self, threshold=5):
        z_scores = stats.zscore(self.rest_data)
        outliers_mask = (z_scores > threshold) | (z_scores < -threshold)
        self.__rest_data = self.rest_data.where(~outliers_mask, other=pd.NA)

        z_scores = stats.zscore(self.rest_train_data)
        outliers_mask = (z_scores > threshold) | (z_scores < -threshold)
        self.__rest_train_data = self.rest_train_data.where(~outliers_mask, other=pd.NA)

    def apply_operation(self, op: Union[str, list[str]]):
        if isinstance(op, str):
            op = [op]
        for instance in op:
            if instance not in ['trend', 'seasonality']:
                warnings.warn(f"{instance} operation is not implemented. Enter operations in ['trend', 'seasonality'].")
        if 'trend' in op:
            self.operation_train.trend(self.train_data)
            self.operation_data.trend(self.data)
        if 'seasonality' in op:
            if 'seasonality' not in self.tests_stat:
                self.apply_stat_test('seasonality')
            self.operation_train.season(self.train_data, self.tests_stat['seasonality'][1])
            self.operation_data.season(self.data, self.tests_stat['seasonality'][1])
        self.__rest_train_data += self.operation_train.apply(-len(self.train_data))
        self.__rest_data += self.operation_data.apply(-len(self.data))

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
        if self.operation_data.list:
            inverse_operations = self.operation_data.unapply(horizon)
        if model_name == 'AggregatedModel':
            if 'AggregatedModel' not in self.models.keys():
                raise Exception('Aggregated Model has not been trained. Use method apply_aggregated_model first.')
            for model in self.models['AggregatedModel']['models'].keys():
                self.models[model]['final_estimator'] = Model(self).compute_final_estimator(model)
                self.models[model]['forecast'] = self.models[model]['final_estimator'].predict(horizon)
                if self.operation_data.list:
                    self.models[model]['forecast'] += TimeSeries.from_dataframe(inverse_operations)

            self.models['AggregatedModel']['forecast'] = AggregatedModel(self).compute_final_estimator()
        else:
            if model_name not in self.models.keys():
                raise Exception(f'{model_name} has not been trained.')
            self.models[model_name]['final_estimator'] = Model(self).compute_final_estimator(model_name)
            self.models[model_name]['forecast'] = self.models[model_name]['final_estimator'].predict(horizon)
            if self.operation_data.list:
                self.models[model_name]['forecast'] += TimeSeries.from_dataframe(inverse_operations)
