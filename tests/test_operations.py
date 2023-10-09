import numpy as np
from darts.datasets import AirPassengersDataset

from pasts.operations import Trend


def test_trend():
    trend = Trend()
    series = AirPassengersDataset().load()
    self = trend
    dataframe = series.pd_dataframe()
    dataframe["#Passengers2"] = dataframe["#Passengers"]
    dataframe["#Passengers2"] *= 100
    i = 10
    trend.fit(X=dataframe)
    assert trend.transform(i).shape[0] == i
    assert np.abs(trend.coef_[0] * 100 - trend.coef_[1]) < 0.001
    assert len(trend.coef_) == dataframe.shape[1]
    assert len(trend.intercept_) == dataframe.shape[1]
    assert all(dataframe.columns == trend.transform(10).columns)
