# Copyright 2023 Eurobios
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import pandas as pd
from darts import TimeSeries
import warnings
from sklearn.metrics import r2_score, mean_squared_error
from darts.metrics import mape, smape, mae
from pasts.statistical_tests import check_arguments


dict_metrics_sklearn = {'r2': r2_score,
                        'mse': mean_squared_error,
                        'rmse': mean_squared_error}

dict_metrics_darts = {'mape': mape,
                      'smape': smape,
                      'mae': mae}


class Metrics:
    """
    A class to compute scores for time series predictions.

    Attributes
    ----------
    signal : Signal
        The signal on which predictions have been computed.
    dict_metrics_sklearn: dict
        Metrics from sklearn requested in class instantiation.
        keys: names in ['r2', 'mse', 'rmse']
        values: functions
    dict_metrics_darts: dict
        Metrics from darts requested in class instantiation.
        keys: names in ['mape', 'smape', 'mae']
        values: functions

    Methods
    -------
    scores_sklearn(model, axis):
        Computes sklearn scores given by dict_metrics_sklearn.
    scores_darts(model):
        Computes darts scores given by dict_metrics_sklearn.
    compute_scores(model, axis):
        Computes all requested scores.
    scores_comparison(axis):
        Creates a dataframe for each scorer with the performance of all models.

    See also
    --------
    pasts.signal for details on the Signal object.
    """

    def __init__(self, signal: "Signal", list_metrics: list[str]):
        """
        Constructs all the necessary attributes for the metrics object.

        Parameters
        ----------
        signal : Signal
            Used for predictions.
        list_metrics: list[str]
            List of names of metrics to be computed.
        """
        self.signal = signal
        self.dict_metrics_sklearn = {key: dict_metrics_sklearn[key] for key in list_metrics
                                     if key in dict_metrics_sklearn.keys()}
        self.dict_metrics_darts = {key: dict_metrics_darts[key] for key in list_metrics
                                   if key in dict_metrics_darts.keys()}

    def _scores_sklearn(self, model: str, axis: int) -> pd.DataFrame:
        """
        Computes sklearn scores given by dict_metrics_sklearn.

        Parameters
        ----------
        model : str
                Model for which to compute the score.
        axis : int = 0 or 1 (default 1)
                Whether to compute scores time-wise (0) or unit-wise (1)

        Returns
        -------
        Dataframe of scores with unit or time as index and metrics as columns
        """
        df_test = self.signal.test_data.copy()
        df_pred = self.signal.models[model]['predictions'].pd_dataframe()

        if axis == 0:
            df_pred = df_pred.transpose()
            df_test = df_test.transpose()

        results = pd.DataFrame(index=df_pred.columns, columns=list(self.dict_metrics_sklearn.keys()))
        for col in df_pred.columns:
            df_temp = pd.DataFrame(df_pred[col])
            df_temp['test'] = df_test[col]
            for metric in self.dict_metrics_sklearn.keys():
                if df_temp.isnull().sum().sum() != 0:
                    warnings.warn('Test set or predictions contain NaN values: they are deleted to compute the metrics',
                                  UserWarning)
                df_temp.dropna(inplace=True)
                test = df_temp[df_temp.columns[0]].values
                pred = df_temp[df_temp.columns[1]].values
                if metric == 'rmse':
                    results.loc[col, metric] = self.dict_metrics_sklearn[metric](test, pred, squared=False)
                else:
                    results.loc[col, metric] = self.dict_metrics_sklearn[metric](test, pred)

        return results

    def _scores_darts(self, model: str) -> pd.DataFrame:
        """
        Computes sklearn scores given by dict_metrics_sklearn.

        Parameters
        ----------
        model : str
                Model for which to compute the score.

        Returns
        -------
        Dataframe of scores with unit or time as index and metrics as columns
        """
        ts_test = TimeSeries.from_dataframe(self.signal.test_data)
        ts_pred = self.signal.models[model]['predictions']
        results = pd.DataFrame(index=ts_pred.columns, columns=list(self.dict_metrics_darts.keys()))
        for col in ts_pred.columns:
            for metric in self.dict_metrics_darts.keys():
                df_test = ts_test.pd_dataframe()
                df_pred = ts_pred.pd_dataframe()
                zero_indices_test = df_test.index[df_test.eq(0.0).any(axis=1)]
                zero_indices_pred = df_pred.index[df_pred.eq(0.0).any(axis=1)]
                zero_indices = zero_indices_test.union(zero_indices_pred)
                if (metric == 'mape') & (len(zero_indices) > 0):
                    warnings.warn('Test set or predictions contain 0 values: cannot compute mape',
                                  UserWarning)
                else:
                    results.loc[col, metric] = self.dict_metrics_darts[metric](ts_test[col], ts_pred[col])
        return results

    def compute_scores(self, model: str, axis: int) -> pd.DataFrame:
        """
        Computes all scores given in class instantiation.

        Parameters
        ----------
        model : str
                Model for which to compute the score.
        axis : int = 0 or 1 (default 1)
                Whether to compute scores time-wise (0) or unit-wise (1)

        Returns
        -------
        Dataframe of scores with unit or time as index and metrics as columns
        """
        check_arguments(
            not self.signal.models,
            "Apply at least one model before computing scores.",
        )

        check_arguments(
            model not in self.signal.models.keys(),
            f"Model {model} has not been applied to this signal.",
        )

        if (axis == 0) & (self.signal.properties['is_univariate']):
            warnings.warn('For univariate time series, scores computed for each date might not be interpretable.')
            if 'r2' in self.dict_metrics_sklearn:
                warnings.warn('R2 cannot be computed.')
                del self.dict_metrics_sklearn['r2']

        df_sk = self._scores_sklearn(model, axis)
        if axis == 0:
            warnings.warn('Only R2, MSE and RMSE can be computed for each date.')
            result = df_sk
        else:
            df_darts = self._scores_darts(model)
            result = pd.concat([df_sk, df_darts], axis=1)
        return result

    def scores_comparison(self, axis: int) -> dict:
        """
        Creates a dataframe for each scorer with the performance of all models.
        Makes model comparison easier.

        Parameters
        ----------
        axis : int = 0 or 1 (default 1)
                Whether to compute scores time-wise (0) or unit-wise (1)

        Returns
        -------
        Dictionary of scores :
            keys: names of metrics
            values: pd.Dataframe of scores with unit or time as index and models as columns
        """
        if axis == 1:
            score_type = 'unit_wise'
        elif axis == 0:
            score_type = 'time_wise'

        dict_metrics_comp = {}
        mod = list(self.signal.models.keys())[0]
        ref = self.signal.models[mod]['scores'][score_type]
        for metric in ref.columns:
            df = pd.DataFrame(index=ref.index, columns=list(self.signal.models.keys()))
            for model in df.columns:
                df[model] = self.signal.models[model]['scores'][score_type][metric]
            dict_metrics_comp[metric] = df
        return dict_metrics_comp
