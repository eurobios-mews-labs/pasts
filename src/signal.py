import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot


class Signal:

    def __init__(self, data: pd.DataFrame) -> None:
        """

        :type data: data contained a times serie in columns and date time in index
        """
        self.data = data
        if not type(data.index) == pd.core.indexes.datetimes.DatetimeIndex:
            print("Error in type of index, should be datetimes")
            exit()
        self.time_step = (self.data.index[1] - self.data.index[0])

    def profiling(self, head=5):
        print("##################### Shape #####################")
        print(self.data.shape)
        print("##################### Types #####################")
        print(self.data.dtypes)
        print("##################### Head #####################")
        print(self.data.head(head))
        print("##################### Tail #####################")
        print(self.data.tail(head))
        print("##################### NA #####################")
        print(self.data.isnull().sum())
        print("##################### Quantiles #####################")
        print(self.data.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

    def plot_st(self):
        self.data.plot(figsize=(20, 10), linewidth=5, fontsize=20)
        plt.xlabel('Year', fontsize=20)
