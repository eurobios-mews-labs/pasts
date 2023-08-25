import copy
import warnings
from abc import ABC

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
    models : dict
        keys: models applied on the series
        values: dictionary of predictions and best parameters

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
        self.__operation_train = None
        self.__operation_data = None
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
        """Input time series as a pandas dataframe"""
        return self.__data

    @property
    def rest_data(self):
        """Residual after applying operations to the data"""
        return self.__rest_data

    @property
    def operation_data(self):
        return self.__operation_data

    @property
    def operation_train(self):
        return self.__operation_train

    @property
    def train_data(self):
        """Train set as a pandas dataframe"""
        return self.__train_data

    @property
    def test_data(self):
        """Test set as a pandas dataframe"""
        return self.__test_data

    @property
    def rest_train_data(self):
        """Residual after applying operations to train set"""
        return self.__rest_train_data

    @property
    def properties(self):
        """Dictionary of properties of the signal"""
        return self.__properties

    @property
    def tests_stat(self):
        """Dictionary of statistical tests applied on data"""
        return self.__tests_stat

    @property
    def performance_models(self):
        """Dictionary containing a maximum of 2 other dictionaries for unit-wise or time-wise scores. Dictionaries
        contain a dataframe for each scorer, with scores computed for all models."""
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

        Fills the attributes train_data, test_data and rest_train_data (with train_data per default)

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
        """
        Deletes outliers in transformed data and train data.

        Parameters
        ----------
        threshold: int
            z-score above which values are deleted.

        Returns
        -------
        None
        """
        z_scores = stats.zscore(self.rest_data)
        outliers_mask = (z_scores > threshold) | (z_scores < -threshold)
        self.__rest_data = self.rest_data.where(~outliers_mask, other=pd.NA)

        z_scores = stats.zscore(self.rest_train_data)
        outliers_mask = (z_scores > threshold) | (z_scores < -threshold)
        self.__rest_train_data = self.rest_train_data.where(~outliers_mask, other=pd.NA)

    def apply_operations(self, list_op: list[str]):
        self.__operation_data = Operation(self.data)
        self.__rest_data = self.operation_data.fit_transform(list_op)
        if self.train_data is not None:
            self.__operation_train = Operation(self.train_data)
            self.__rest_train_data = self.operation_train.fit_transform(list_op)

    def apply_model(self, model, gridsearch=False, parameters=None):
        """
        Applies statistical, machine learning of deep learning model to the series.

        Fills the attribute models.

        Parameters
        ----------
        model
                Instance of a model. Will be refitted even if it has already been.
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
                Name of a model. If AggregatedModel, forecasts will be computed with the models included in the
                aggregation.
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
                if self.operation_data.dict_op:
                    self.models[model]['forecast'] = TimeSeries.from_dataframe(
                        self.operation_data.transform(self.models[model]['forecast'].pd_dataframe()))

            self.models['AggregatedModel']['forecast'] = AggregatedModel(self).compute_final_estimator()
        else:
            if model_name not in self.models.keys():
                raise Exception(f'{model_name} has not been trained.')
            self.models[model_name]['final_estimator'] = Model(self).compute_final_estimator(model_name)
            self.models[model_name]['forecast'] = self.models[model_name]['final_estimator'].predict(horizon)
            if self.operation_data.dict_op:
                self.models[model_name]['forecast'] = TimeSeries.from_dataframe(
                    self.operation_data.transform(self.models[model_name]['forecast'].pd_dataframe()))
