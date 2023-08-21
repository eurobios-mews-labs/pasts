import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


class Validation:

    def __init__(self, data: pd.DataFrame):
        self.__cv_tseries = None
        self.__data = data
        self.train_data = None
        self.test_data = None

    @property
    def cv_tseries(self):
        return self.__cv_tseries

    def split_cv(self, timestamp, n_splits_cv=None):
        self.train_data = self.__data.loc[self.__data.index <= timestamp]
        self.test_data = self.__data.loc[self.__data.index > timestamp]

        print("Split applied on :", timestamp, '\n')

        if n_splits_cv is not None:
            time_series_cross_validation = TimeSeriesSplit(n_splits=n_splits_cv)

            for fold, (train_index, test_index) in enumerate(time_series_cross_validation.split(self.train_data)):
                print("Fold: {}".format(fold))
                print("TRAIN indices:", train_index[0], " -->", train_index[-1])
                print("TEST  indices:", test_index[0], "-->", test_index[-1])
                print("\n")

            self.__cv_tseries = time_series_cross_validation.split(self.train_data)
