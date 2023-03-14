import pandas as pd

from seriestemporelles.signal.proporties_signal import Proprieties, dict_test_uni_variate, dict_test_multi_variate


def check_arguments(
        condition: bool,
        message: str = ""):
    """
    Checks  boolean condition and raises a ValueError.

    """

    if condition:
        raise TypeError(message)


class TestStatistics:

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.proprieties = Proprieties(self.data)

    def statistical_test(self, type_test: str, test_stat_name: str,
                         *args, **kwargs) -> object:
        if self.proprieties:
            dict_test = dict_test_uni_variate
        else:
            dict_test = dict_test_multi_variate

        check_arguments(not self.proprieties.is_uni_variate,
                        "Only uni_variate time series can be tested")

        check_arguments(
            type_test not in dict_test.keys(),
            f"Select the type of statistical test from: {dict_test.keys()}.",
        )

        check_arguments(
            test_stat_name not in dict_test[type_test],
            f"Select correct test from: {dict_test[type_test]}.",
        )
        if type_test == 'stationary':
            return self.proprieties.is_stationary(test_name=test_stat_name, *args, **kwargs)
        elif type_test == 'seasonality':
            return self.proprieties.is_seasonality(*args, **kwargs)
        else:
            return self.proprieties.causality(*args, **kwargs)
