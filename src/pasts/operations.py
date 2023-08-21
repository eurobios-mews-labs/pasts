import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.models import ExponentialSmoothing, AutoARIMA, Prophet
from darts.utils.utils import ModelMode, SeasonalityMode
from statsmodels.tsa.seasonal import seasonal_decompose

from pasts.signal import Signal
from pasts.visualization import Visualisation


class DecomposedSignal(Signal):

    def __init__(self, data):
        super().__init__(data)
        self.detrend_signals = pd.DataFrame(np.zeros((len(self.data.index), len(self.data.columns))),
                                            columns=self.data.columns, index=self.data.index)
        self.trend_linear = pd.DataFrame(np.zeros((len(self.data.index), len(self.data.columns))),
                                         columns=self.data.columns, index=self.data.index)
        self.trend_periodic = pd.DataFrame(np.zeros((len(self.data.index), len(self.data.columns))),
                                           columns=self.data.columns, index=self.data.index)
        self.data_raw = pd.DataFrame(np.zeros((len(self.data.index), len(self.data.columns))),
                                     columns=self.data.columns, index=self.data.index)
        self.got_trend = 0

    def detrend(self):
        self.detrend_signals = pd.DataFrame(np.zeros((len(self.data.index), len(self.data.columns))),
                                            columns=self.data.columns, index=self.data.index)
        self.trend_linear = pd.DataFrame(np.zeros((len(self.data.index), len(self.data.columns))),
                                         columns=self.data.columns, index=self.data.index)
        self.trend_periodic = pd.DataFrame(np.zeros((len(self.data.index), len(self.data.columns))),
                                           columns=self.data.columns, index=self.data.index)
        self.data_raw = pd.DataFrame(np.zeros((len(self.data.index), len(self.data.columns))),
                                     columns=self.data.columns, index=self.data.index)
        for i in self.data.columns:
            trend_linear = seasonal_decompose(self.data[i], model='additive').trend
            for j in np.arange(-6, 0):
                trend_linear[j] = (2 * trend_linear[j - 1] - trend_linear[j - 2])
                trend_linear[-j - 1] = (2 * trend_linear[-j] - trend_linear[-j + 1])
            self.trend_periodic[i] = seasonal_decompose(self.data[i], model='additive').seasonal
            self.trend_linear[i] = trend_linear
            self.data_raw[i] = list(self.data[i])
            self.data[i] = self.data[i] - self.trend_linear[i] - self.trend_periodic[i]
        self.got_trend = 1

    def retrend_uni(self, fc):
        type_output = type(fc).__name__
        if self.got_trend == 1:
            if type_output == 'TimeSeries':
                fc = fc.pd_dataframe()
            else:
                fc = fc.copy()
            for k in fc.columns:
                period = self.trend_periodic[0:12]
                period.index = [t.strftime("%m") for t in period.index]
                step = self.trend_linear.iloc[-1] - self.trend_linear.iloc[-2]
                for t, val in enumerate(fc[k].copy()):
                    if fc.index[t] in self.trend_linear.index:
                        fc[k][t] = val + self.trend_linear.loc[fc.index[t]][0].copy() + self.trend_periodic.loc[fc.index[t]][0].copy()
                    else:
                        ID = fc[k].index[t].strftime("%m")
                        fc[k][t] = fc[k][t] + t * step + period.loc[ID]
            if type_output == "TimeSeries":
                fc = TimeSeries.from_dataframe(fc)
        return fc

    def retrend_multi(self, fc):
        if self.got_trend == 1:
            for k in fc.columns:
                period = self.trend_periodic[k][0:12]
                period.index = [t.strftime("%m") for t in period.index]
                step = self.trend_linear[k].iloc[-1] - self.trend_linear[k].iloc[-2]
                for t, val in enumerate(fc[k]):
                    if fc.index[t] in self.trend_linear[k].index:
                        fc[k][t] = val + self.trend_linear[k].loc[fc.index[t]] + self.trend_periodic[k].loc[fc.index[t]]
                    else:
                        ID = fc[k].index[t].strftime("%m")
                        fc[k][t] = fc[k][t] + t * step + period.loc[ID]
        return fc
