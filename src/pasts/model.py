from abc import ABC, abstractmethod
from pandas import MultiIndex

from pasts.metrics import root_mean_squared_error

import pandas as pd
from darts import TimeSeries


class ModelAbstract(ABC):
    def __init__(self, signal: "Signal"):
        self.signal = signal

    @abstractmethod
    def apply(self):
        pass

    @abstractmethod
    def compute_final_estimator(self):
        pass


class Model(ModelAbstract):

    def __init__(self, signal: "Signal"):
        super().__init__(signal)

    def apply(self, model, gridsearch=False, parameters=None):
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

        if self.signal.operation_train.dict:
            forecast = TimeSeries.from_dataframe(self.signal.operation_train.tranform(forecast.pd_dataframe()))

        return {'predictions': forecast, 'best_parameters': best_parameters, 'scores': {'unit_wise': {},
                                                                                        'time_wise': {}},
                'estimator': model}

    def compute_final_estimator(self, model_name):
        if model_name not in self.signal.models.keys():
            raise AttributeError(f'{model_name} has not been fitted.')
        model = self.signal.models[model_name]['estimator']
        train_temp = TimeSeries.from_dataframe(self.signal.rest_data)
        model.fit(train_temp)
        return model


class AggregatedModel(ModelAbstract):

    def __init__(self, signal: "Signal"):
        super().__init__(signal)

    def apply(self, dict_models):
        dict_pred = {model: self.signal.models[model][
            'predictions'].pd_dataframe().copy() for model in dict_models.keys()}
        df_test = self.signal.test_data.copy()
        weights = pd.DataFrame(index=MultiIndex.from_product([self.signal.test_data.index,
                                                             self.signal.test_data.columns], names=['Date', 'Unité']),
                               columns=list(dict_models.keys()))
        weights.drop(self.signal.test_data.index[0], level=0, inplace=True)
        for model in weights.columns:
            df_pred = dict_pred[model]
            for date in weights.index.get_level_values(0).unique():
                df_pred_temp = df_pred[df_pred.index < date]
                df_test_temp = df_test[df_test.index < date]
                for ref in weights.index.get_level_values(1).unique():
                    weights.loc[(date, ref)][model] = 1 / root_mean_squared_error(df_test_temp[ref], df_pred_temp[ref])
        for i in weights.index:
            weights.loc[i] = weights.loc[i] / (weights.loc[i].sum())
        weights = weights.groupby('Unité')[list(dict_models.keys())].mean()

        df_ag = pd.DataFrame(index=self.signal.models[list(dict_models.keys())[0]]['predictions'].time_index,
                             columns=self.signal.models[list(dict_models.keys())[0]]['predictions'].columns)
        for ref in df_ag.columns:
            res = [0 for i in df_ag.index]
            for model in dict_models.keys():
                res += self.signal.models[model]['predictions'][ref].values() * weights.loc[ref, model]
            df_ag[ref] = res
        return {'predictions': TimeSeries.from_dataframe(df_ag), 'weights': weights, 'models': dict_models,
                'scores': {}}

    def compute_final_estimator(self):
        dict_models = self.signal.models['AggregatedModel']['models']
        df_ag = pd.DataFrame(index=self.signal.models[list(dict_models.keys())[0]]['forecast'].time_index,
                             columns=self.signal.models[list(dict_models.keys())[0]]['forecast'].columns)
        for ref in df_ag.columns:
            res = [0 for i in df_ag.index]
            for model in dict_models.keys():
                res += self.signal.models[model]['forecast'][ref].values() * self.signal.models['AggregatedModel'][
                    'weights'].loc[ref, model]
            df_ag[ref] = res

        return TimeSeries.from_dataframe(df_ag)
