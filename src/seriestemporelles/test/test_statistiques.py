import pandas as pd
from darts.utils.statistics import check_seasonality
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss

dict_test = {'stationary': [kpss, adfuller],
             # 'causality': [grangercausalitytests],
             'seasonality': [check_seasonality]
             }


class TestStatistics:

    def __init__(self, data: pd.DataFrame, ):
        self.data = data

    def statistical_test(self, type_test, test_stat, dict_test=dict_test,
                         *args, **kwargs) -> object:
        if type_test not in dict_test.keys():
            raise ValueError("Select the type of statistical test from", dict_test.keys())
        else:
            if test_stat not in dict_test[type_test]:
                raise TypeError("Select correct test from", dict_test[type_test])
            else:
                test = test_stat(self.data, *args, **kwargs)
                # test_output = pd.Series(test[0:3], index=['Test Statistic', 'p-value', '#Lags Used'])
        return test

# to dod
# 1 preciser si le test et dans quelle cas et unifier les
# résultats sous forme d'un tuple de 3 éléments
# (test_name, val, p_value) avec p_value==0.05 par défaut
# en envoi une liste des  tuple pour touts les résultats)
