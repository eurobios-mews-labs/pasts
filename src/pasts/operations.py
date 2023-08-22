import pandas as pd
from darts.utils.historical_forecasts.utils import TimeIndex
from sklearn.linear_model import LinearRegression


class Identity:
    def transform(self, t: TimeIndex):
        return t

    def back_transform(self, t: TimeIndex):
        return t


class Operation:
    def __init__(self, signal: "Signal"):
        self.list = [(Identity().transform(signal.data.index), Identity().back_transform(signal.data.index))]
