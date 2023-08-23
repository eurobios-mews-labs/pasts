import numpy as np
import pandas as pd
from darts.utils.historical_forecasts.utils import TimeIndex
from sklearn.linear_model import LinearRegression


class Trend:
    def __init__(self):
        ...

    def fit(self, X, y=None):
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

    def _t(self, i):
        t = self._from_i_to_vector(i)
        t = t.reshape(1, -1) * np.ones((
            self.coef_.shape[0],
            np.abs(i)))
        return t

    def _intercept_to_frame(self, shape):
        return self.intercept_.reshape(-1, 1)* np.ones((
            self.coef_.shape[0],
            shape))

    def transform(self, i, y=None):
        t = self._t(i)
        b = self._intercept_to_frame(t.shape[1])
        ret = self.coef_ * t + b
        return pd.DataFrame(ret,
                            columns=pd.to_datetime(t[0, :].ravel()*1e17),
                            index=self.features_.to_list()).T

    def reverse_transform(self, i, y=None):
        t = self._t(i)
        b = self._intercept_to_frame(t.shape[1])
        ret = - self.coef_ * t - b
        return pd.DataFrame(ret,
                            columns=pd.to_datetime(t[0, :].ravel() * 1e17),
                            index=self.features_.to_list()).T


class Identity:
    def transform(self, t: TimeIndex):
        return t

    def back_transform(self, t: TimeIndex):
        return t


class Operation:
    def __init__(self, signal: "Signal"):
        self.list = [(Identity().transform(signal.data.index),
                      Identity().back_transform(signal.data.index))]


if __name__ == '__main__':
    # TODO to del
    from darts.datasets import AirPassengersDataset
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("qt5agg")
    series = AirPassengersDataset().load()
    dataframe = series.pd_dataframe()
    dataframe["#Passengers2"] = dataframe["#Passengers"]
    dataframe["#Passengers2"] *= 1.5
    trend = Trend()
    trend.fit(X=dataframe)
    ret = trend.transform(10)
    ret2 = trend.transform(-100)

    plt.figure()
    ax = plt.gca()
    series.plot(ax=ax)
    ret.plot(ax=ax)
    ret2.plot(ax=ax)
    plt.show()

    ret_past = trend.reverse_transform(-100)
    ret_past.plot(ax=ax)
    ret_past.index = ret_past.index.to_series().dt.ceil("ms")
    (ret_past + dataframe).plot(label="detrend", ax=ax)
    plt.legend()