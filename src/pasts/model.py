# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

from abc import ABC, abstractmethod

import numpy as np
from pandas import MultiIndex

import pandas as pd
from darts import TimeSeries
from sklearn.metrics import mean_squared_error


class ModelAbstract(ABC):
    """
    An abstract class to represent a forecasting model.

    Attributes
    ----------
    signal : Signal
        Object on which to apply a model

    Methods
    -------
    apply():
        Applies a model.
    compute_final_estimator():
        Fits the model on whole dataset.
    """
    def __init__(self, signal: "Signal"):
        """
        Constructs all the necessary attributes for the model object.

        Parameters
        ----------
        signal : Signal
            Object on which to apply models.
        """
        self.signal = signal

    @abstractmethod
    def apply(self, model, gridsearch, parameters):
        pass

    @abstractmethod
    def compute_final_estimator(self, model_name: str):
        pass


class Model(ModelAbstract):

    def __init__(self, signal: "Signal"):
        """
        Constructs all the necessary attributes for the model object.

        Parameters
        ----------
        signal : Signal
                Object on which to apply models.
        """
        super().__init__(signal)

    def apply(self, model: object, gridsearch: bool = False, parameters: dict = None) -> dict:
        """
        Applies given model on test set.
        If gridsearch is True and parameters are given, performs a gridsearch and saves the best parameters.

        Parameters
        ----------
        model : instance of a model from darts
               Model to apply
        gridsearch : bool, optional
               Whether to perform a gridsearch (default is False)
        parameters : dict
                Parameters used to perform the gridsearch (default is None)
                keys: names of parameters
                values: lists of parameters to test

        Returns
        -------
        {'predictions' : predicted TimeSeries,
        'best_parameters' : parameters selected by gridsearch, "default" if no gridsearch,
        'scores' : {'unit_wise': {}, 'time_wise': {}} will be filled when scores are computed,
        'estimator' : saves the trained estimator}

        See also
        --------
        darts.models to see available models.
        """
        train_tseries = TimeSeries.from_dataframe(self.signal.rest_train_data)
        if gridsearch:
            if parameters is None:
                raise Exception("Please enter the parameters")
            print('Performing the gridsearch for', model.__class__.__name__, '...')
            best_model, best_parameters, _ = model.gridsearch(parameters=parameters,
                                                              series=train_tseries,
                                                              start=0.5,
                                                              forecast_horizon=5)
            model = best_model
        else:
            best_parameters = "default"

        model.fit(train_tseries)
        forecast = model.predict(len(self.signal.test_data))
        if self.signal.operation_train is not None:
            if self.signal.operation_train.dict_op:
                forecast = TimeSeries.from_dataframe(self.signal.operation_train.transform(forecast.pd_dataframe()))

        return {'predictions': forecast, 'best_parameters': best_parameters, 'scores': {'unit_wise': {},
                                                                                        'time_wise': {}},
                'estimator': model}

    def compute_final_estimator(self, model_name: str):
        """
        Fits given model on whole dataset. Requires the model to have been fitted on train set.

        Parameters
        ----------
        model_name : str
               Name of model to fit

        Returns
        -------
        Fitted estimator
        """
        if model_name not in self.signal.models.keys():
            raise AttributeError(f'{model_name} has not been fitted.')
        model = self.signal.models[model_name]['estimator']
        train_temp = TimeSeries.from_dataframe(self.signal.rest_data)
        model.fit(train_temp)
        return model


class AggregatedModel(ModelAbstract):

    def __init__(self, signal: "Signal"):
        """
        Constructs all the necessary attributes for the model object.

        Parameters
        ----------
        signal : Signal
                Object on which to apply models.
        """
        super().__init__(signal)

    def apply(self, model: dict, gridsearch=None, parameters=None) -> dict:
        """
        Aggregates given models according to their performance on test set.
        Requires the models to have been applied on test set.

        Parameters
        ----------
        model : dict
               keys: model names
               values: model instances
        gridsearch : Unused, added for compatibility
        parameters : Unused, added for compatibility

        Returns
        -------
        {'predictions' : predicted TimeSeries,
        'weights' : Dataframe of weights for each model,
        'models' : stores passed dict_models,
        'scores' : will be filled when scores are computed}
        """
        dict_pred = {model: self.signal.models[model][
            'predictions'].pd_dataframe().copy() for model in model.keys()}
        df_test = self.signal.test_data.copy()
        weights = pd.DataFrame(index=MultiIndex.from_product([self.signal.test_data.index,
                                                             self.signal.test_data.columns], names=['Date', 'Unité']),
                               columns=list(model.keys()))
        weights.drop(self.signal.test_data.index[0], level=0, inplace=True)
        for model_ in weights.columns:
            df_pred = dict_pred[model_]
            for date in weights.index.get_level_values(0).unique():
                df_pred_temp = df_pred[df_pred.index < date]
                df_test_temp = df_test[df_test.index < date]
                for ref in weights.index.get_level_values(1).unique():
                    weights.loc[(date, ref)][model_] = 1 / mean_squared_error(df_test_temp[ref], df_pred_temp[ref],
                                                                              squared=False)

        for i in weights.index:
            weights.loc[i] = weights.loc[i] / (weights.loc[i].sum())
        weights = weights.groupby('Unité')[list(model.keys())].mean()

        df_ag = pd.DataFrame(index=self.signal.models[list(model.keys())[0]]['predictions'].time_index,
                             columns=self.signal.models[list(model.keys())[0]]['predictions'].columns)

        for ref in df_ag.columns:
            res = [0 for _ in df_ag.index]
            for model_ in model.keys():
                pred = self.signal.models[model_]['predictions'].pd_dataframe()[ref].values
                res += pred * weights.loc[ref, model_]
            df_ag[ref] = res

        return {'predictions': TimeSeries.from_dataframe(df_ag), 'weights': weights,
                'models': model, 'scores': {'unit_wise': {}, 'time_wise': {}}}

    def compute_final_estimator(self, model_name=''):
        """
        Aggregates all computed forecasts.

        Parameters
        ----------
        model_name : Unused, added for compatibility

        Returns
        -------
        Predicted TimeSeries
        """
        dict_models = self.signal.models['AggregatedModel']['models']
        df_ag = pd.DataFrame(index=self.signal.models[list(dict_models.keys())[0]]['forecast'].time_index,
                             columns=self.signal.models[list(dict_models.keys())[0]]['forecast'].columns)
        # conf_itv = df_ag.copy()
        # std = self.signal.models['AggregatedModel']['std_test']
        for ref in df_ag.columns:
            res = [0 for _ in df_ag.index]
            # itv_inf = [0 for _ in df_ag.index]
            # itv_sup = [0 for _ in df_ag.index]
            for model_ in dict_models.keys():
                pred = self.signal.models[model_]['forecast'].pd_dataframe()[ref].values
                res += pred * self.signal.models['AggregatedModel']['weights'].loc[ref, model_]
                # itv_inf = [itv_inf[i] + self.signal.models['AggregatedModel']['weights'].loc[ref, model_] * (
                #             pred[i] + (-1.96) * std[model_] * np.sqrt(i)) for i in range(len(itv_inf))]
                # itv_sup = [itv_sup[i] + self.signal.models['AggregatedModel']['weights'].loc[ref, model_] * (
                #             pred[i] + 1.96 * std[model_] * np.sqrt(i)) for i in range(len(itv_sup))]
            df_ag[ref] = res
            # conf_itv[ref] = [[itv_inf[i], itv_sup[i]] for i in range(len(itv_inf))]

        return TimeSeries.from_dataframe(df_ag)  # , conf_itv
