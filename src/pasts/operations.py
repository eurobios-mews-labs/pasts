import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.historical_forecasts.utils import TimeIndex
from sklearn.linear_model import LinearRegression


class Trend:
    def __init__(self):
        ...

    def fit(self, X, y=None):
        self.origin_index = X.index
        self.time_index = X.index.to_numpy(dtype="float64")
        self.time_index = self.time_index.astype(float)#*1e-17
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


class Identity:
    def transform(self, t: TimeIndex):
        return t

    def back_transform(self, t: TimeIndex):
        return t


class Operation:
    def __init__(self):
        # self.list = [(Identity().transform(signal.data.index),
        #              Identity().back_transform(signal.data.index))]
        self.list = []

    def trend(self, data):
        cls = Trend()
        cls.fit(data)
        self.list.append((cls.transform, cls.reverse_transform))

    def apply(self, i):
        dataframes = []
        for op in self.list:
            dataframes.append(op[0](i))
        result = dataframes[0]
        for i in range(1, len(dataframes)):
            result += dataframes[i]
        return result

    def unapply(self, i):
        dataframes = []
        for op in self.list:
            dataframes.append(op[1](i))
        result = dataframes[0]
        for i in range(1, len(dataframes)):
            result += dataframes[i]
        return result

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
    op = Operation()
    op.trend(dataframe)
    ret = op.unapply(50)
    ret2 = op.unapply(-100)

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


