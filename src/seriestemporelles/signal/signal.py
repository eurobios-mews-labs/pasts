import copy
import warnings
from abc import ABC

import pandas as pd

from seriestemporelles.signal.model import Model, SimpleModel, AggregatedModel
from seriestemporelles.signal.test_statistiques import TestStatistics, dict_test
from seriestemporelles.signal.validation import Validation
from seriestemporelles.signal.metrics import Metrics


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
    data : pd.Dataframe
        Dataframe of time series
    properties : dict
        Info about the series (shape, types, univariate or multivariate, number of missing values, quantiles)
    tests_stat : dict
        keys: names of statistical tests applied on the series
        values: Whether the null hypothesis is rejected, p-value
    train_data : pd.Dataframe
        train set
    test_data : pd.Dataframe
        test set
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
        self.properties = profiling(data)
        self.tests_stat = {}
        self.__call_test = TestStatistics(self)
        self.train_data = None
        self.test_data = None
        self.__cv_tseries = None
        self.models = {}
        self.performance_models = None

    @property
    def data(self):
        return self.__data

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

        if (type_test == 'stationary') & (test_stat_name is None):
            test_stat_name = 'adfuller'
            self.tests_stat[f"{type_test}: {test_stat_name}"] = self.__call_test.apply(type_test, test_stat_name,
                                                                                       *args, **kwargs)
        elif (type_test == 'stationary') & (test_stat_name is not None):
            self.tests_stat[f"{type_test}: {test_stat_name}"] = self.__call_test.apply(type_test, test_stat_name,
                                                                                       *args, **kwargs)
        else:
            test_stat_name = dict_test[type_test]
            self.tests_stat[f"{type_test}: {test_stat_name}"] = self.__call_test.apply(type_test, test_stat_name,
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
        self.train_data = call_validation.train_data
        self.test_data = call_validation.test_data
        self.__cv_tseries = call_validation.cv_tseries

    def apply_model(self, model, gridsearch=False, parameters=None):
        """
        Applies statistical, machine learning of deep learning model to the series.

        Fills the attribute models.

        Parameters
        ----------
        model
                Instance of a model.
        gridsearch : bool, optional
                Whether to perform a gridsearch (default is False)
        parameters : pd.Dataframe, optional
                Parameters to test if a gridsearch is performed (default is None)

        Returns
        -------
        None
        """
        self.models[model.__class__.__name__] = SimpleModel(self).apply(model, gridsearch, parameters)

    def compute_scores(self, list_metrics: list[str] = None, axis=1):
        if list_metrics is None:
            list_metrics = ['r2', 'mse', 'rmse', 'mape', 'smape', 'mae']
        call_metric = Metrics(self, list_metrics)
        for model in self.models.keys():
            self.models[model]['scores'] = call_metric.compute_score(model, axis)
        self.performance_models = call_metric.compare_models()

    def forecast_simple(self, model, horizon):
        self.models[model.__class__.__name__]['forecast'] = SimpleModel(self).forecast(model, horizon)

    def apply_aggregated_model(self, list_models, use_fitted=True):
        dict_models = {model.__class__.__name__: model for model in list_models}
        if not use_fitted:
            for model in dict_models.values():
                self.apply_model(model)
        else:
            for model_name, model in dict_models.items():
                if model_name not in self.models.keys():
                    warnings.warn(f'{model_name}  has not yet been fitted. Fitting {model_name}...', UserWarning)
                    self.apply_model(model)
        self.models['AggregatedModel'] = AggregatedModel(self).apply(dict_models)

    def forecast_aggregated(self, horizon):
        if 'AggregatedModel' not in self.models.keys():
            raise Exception('Aggregated Model has not been trained. Use method apply_aggregated_model first.')
        for model_name, model in self.models['AggregatedModel']['models'].items():
            if 'forecast' not in self.models[model_name]:
                self.forecast_simple(model, horizon)
        self.models['AggregatedModel']['forecast'] = AggregatedModel(self).forecast()



