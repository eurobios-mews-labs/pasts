from abc import ABC

import pandas as pd
from typing import Dict, Any, Union, Tuple
from seriestemporelles.signals.signal_abs import Signals
from seriestemporelles.test.test_statistiques import TestStatistics
from pandas.plotting import autocorrelation_plot
# from sklearn.model_selection import TimeSeriesSplit
# from darts import TimeSeries


class UniSignal(Signals, TestStatistics):

    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.report = super().profiling()

    def is_stationary(self, test_stat, *args, **kwargs):
        test_output = super(UniSignal, self)._is_stationary(test_stat, *args, **kwargs)
        self.report[test_stat.__name__] = test_output
        return self.report

    def acf_plot(self):
        autocorrelation_plot(self.data)

    # def split_cv(self, n_splits):
    #     time_series_cross_validation = TimeSeriesSplit(n_splits=n_splits)
    #     return time_series_cross_validation.split(TimeSeries.from_dataframe(self.times_series))
    #
    # def apply_model(self, model, train_index=[], test_index=[],
    #                 parameters: list = {}, gridsearch: bool = False):
    #     data_train = self.data.iloc[train_index]
    #     data_test = self.data.iloc[test_index]
    #     if gridsearch:
    #         best_model, best_parametres = model.gridsearch(parameters=parameters,
    #                                                        series=data_train,
    #                                                        start=0.5,
    #                                                        forecast_horizon=12)
    #     best_model = model
    #     best_model.fit(data_train)
    #     forecast = model.predict(len(data_test))
    #     print('model {} obtains MAPE: {:.2f}%'.format(model, mape(data_test, forecast)))
