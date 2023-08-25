from typing import Union

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import XGBModel
from darts.utils.historical_forecasts.utils import TimeIndex
from darts.utils.statistics import check_seasonality
from sklearn.linear_model import LinearRegression


class Trend:
    def __init__(self):
        ...

    def fit(self, X, y=None):
        self.origin_index = X.index
        self.time_index = X.index.to_numpy(dtype="float64")
        self.time_index = self.time_index.astype(float)*1e-17
        self.features_ = X.columns
        frame = self.time_index.reshape(-1, 1)
        estimator = LinearRegression()
        estimator.fit(frame, X.values)
        self.coef_ = estimator.coef_
        self.intercept_ = estimator.intercept_
        self.dt_ = np.mean(np.diff(self.time_index))
        self.t0_ = self.time_index[-1]
        return self

    def _from_i_to_vector(self, i):
        if i > 0:
            return self.t0_ + np.array(range(i)) * self.dt_
        else:
            return self.time_index[i:]

    def _from_i_to_time_index(self, i):
        if i > 0:
            time_index = pd.date_range(start=self.origin_index[-1], periods=i+1, freq=self.origin_index.freq)[1:]
        else:
            time_index = self.origin_index[i:]
        return time_index

    def _t(self, i):
        t = self._from_i_to_vector(i)
        t = t.reshape(1, -1) * np.ones((
            self.coef_.shape[0],
            np.abs(i)))
        return t

    def _intercept_to_frame(self, shape):
        return self.intercept_.reshape(-1, 1) * np.ones((
            self.coef_.shape[0],
            shape))

    def reverse_transform(self, i, y=None):
        t = self._t(i)
        b = self._intercept_to_frame(t.shape[1])
        ret = self.coef_ * t + b
        return pd.DataFrame(ret,
                            columns=self._from_i_to_time_index(i),
                            index=self.features_.to_list()).T

    def transform(self, i, y=None):
        t = self._t(i)
        b = self._intercept_to_frame(t.shape[1])
        ret = - self.coef_ * t - b
        return pd.DataFrame(ret,
                            columns=self._from_i_to_time_index(i),
                            index=self.features_.to_list()).T


class Seasonality:
    def __init__(self, seasonality):
        self.seasonality = int(seasonality)

    def fit(self, X):
        self.time_index = X.index
        df_diff = pd.DataFrame(index=self.time_index, columns=X.columns)
        for col in X.columns:
            values = X[col].values
            diff = [0 for _ in range(self.seasonality)]
            for i in range(self.seasonality, len(values)):
                res = values[i - self.seasonality]
                diff.append(res)
            df_diff[col] = diff
        self.seasonal_component = df_diff
        self.estimator_future_season = XGBModel(lags=self.seasonality)
        self.estimator_future_season.fit(TimeSeries.from_dataframe(self.seasonal_component))

    def transform(self, i):
        if i > 0:
            output = self.estimator_future_season.predict(i).pd_dataframe()
        else:
            output = self.seasonal_component.iloc[i:]
        return - output

    def reverse_transform(self, i):
        if i > 0:
            output = self.estimator_future_season.predict(i).pd_dataframe()
        else:
            output = self.seasonal_component.iloc[i:]
        return output


class Operation:
    def __init__(self, train_data: pd.DataFrame):
        self.train_data = train_data
        self.rest_data = train_data.copy()
        self.dict_op = {}
        self.implemented_operations = {'trend': self.trend, 'seasonality': self.season}

    def trend(self):
        cls = Trend()
        cls.fit(self.rest_data)
        self.dict_op['trend'] = (cls.transform, cls.reverse_transform)

    def season(self):
        if self.rest_data.shape[1] > 1:
            raise Exception("Removal of seasonal component is not implemented for multivariate TimeSeries.")
        seasonality = check_seasonality(TimeSeries.from_dataframe(
            self.rest_data))
        if seasonality[0]:
            cls = Seasonality(seasonality[1])
            cls.fit(self.rest_data)
            self.dict_op['seasonality'] = (cls.transform, cls.reverse_transform)

    def fit_transform(self, list_op: list[str]):
        for op in list_op:
            if op not in self.implemented_operations.keys():
                raise Exception(f"Operation {op} is not implemented. "
                                f"Choose operations in {self.implemented_operations.keys()}.")
            self.implemented_operations[op]()
            frame = self.dict_op[op][0](-len(self.rest_data))
            self.rest_data += frame
        return self.rest_data

    def transform(self, data: pd.DataFrame, reverse=True):
        intersection_length = len(self.rest_data.index.intersection(data.index))
        if intersection_length > 0:
            i = -intersection_length
        else:
            i = len(data)
        a = 1
        if not reverse:
            a = 0
        for op in reversed(self.dict_op.values()):
            frame = op[a](i)
            data += frame
        return data


if __name__ == '__main__':
    # TODO to del
    from darts.datasets import AirPassengersDataset, AustralianTourismDataset
    import matplotlib.pyplot as plt
    import matplotlib
    # matplotlib.use("qt5agg")
    series = AirPassengersDataset().load()
    dataframe = series.pd_dataframe()
    dataframe["#Passengers2"] = dataframe["#Passengers"]
    dataframe["#Passengers2"] *= 1.5
    season = Seasonality()
    season.fit(X=dataframe)
    season.transform(20)
    season.seasonal_component.plot()

    trend = Trend()
    trend.fit(X=dataframe)
    ret = trend.transform(20)
    ret2 = trend.transform(-100)

    plt.figure()
    ax = plt.gca()
    series.plot(ax=ax)
    ret.plot(ax=ax)
    ret2.plot(ax=ax)

    ret_past = trend.reverse_transform(-100)
    ret_past.plot(ax=ax)
    ret_past.index = ret_past.index.to_series().dt.ceil("ms")
    (ret_past + dataframe).plot(label="detrend", ax=ax)
    plt.legend()
    plt.show()

    # Multivariate
    series_m = AustralianTourismDataset().load()[['Hol', 'VFR', 'Oth']]
    df_m = pd.DataFrame(series_m.values())
    df_m.rename(columns={0: 'Hol', 1: 'VFR', 2: 'Oth'}, inplace=True)
    df_m.index = pd.date_range(start='2020-01-01', periods=len(df_m), freq='MS')
    trend = Trend()
    trend.fit(X=df_m)
    ret = trend.transform(10)
    ret2 = trend.transform(-20)

    i = 0
    plt.figure()
    ax = plt.gca()
    TimeSeries.from_dataframe(df_m).plot(ax=ax)
    for col in df_m.columns:
        ax.lines[i].set_label(f'actual series {col}')
        i += 1
    ret.plot(ax=ax)
    for col in ret.columns:
        ax.lines[i].set_label(f'trend {col}')
        i += 1
    ret2.plot(ax=ax)
    for col in ret2.columns:
        ax.lines[i].set_label(f'trend {col}')
        i += 1

    ret_past = trend.reverse_transform(-20)
    ret_past.plot(ax=ax)
    for col in ret_past.columns:
        ax.lines[i].set_label(f'detrend {col}')
        i += 1
    ret_past.index = ret_past.index.to_series().dt.round("ms")
    (ret_past + df_m).plot(ax=ax)
    for col in df_m.columns:
        ax.lines[i].set_label(f'detrended series {col}')
        i += 1
    plt.legend()
    plt.show()

    # Test Operation
    series = AirPassengersDataset().load()
    dataframe = series.pd_dataframe()
    op = Operation(dataframe)
    det = op.fit_transform(['trend', 'seasonality'])
    ret = op.transform(det)

    i = 0
    plt.figure()
    ax = plt.gca()
    TimeSeries.from_dataframe(dataframe).plot(ax=ax)
    for col in dataframe.columns:
        ax.lines[i].set_label(f'actual series {col}')
        i += 1
    ret.plot(ax=ax)
    for col in ret.columns:
        ax.lines[i].set_label(f'trend {col}')
        i += 1
    ret2.plot(ax=ax)
    for col in ret2.columns:
        ax.lines[i].set_label(f'trend {col}')
        i += 1

    ret_past = op.apply(-100)
    ret_past.plot(ax=ax)
    for col in ret_past.columns:
        ax.lines[i].set_label(f'detrend {col}')
        i += 1
    ret_past.index = ret_past.index.to_series().dt.round("ms")
    (ret_past + dataframe).plot(ax=ax)
    for col in dataframe.columns:
        ax.lines[i].set_label(f'detrended series {col}')
        i += 1
    plt.legend()
    plt.show()


