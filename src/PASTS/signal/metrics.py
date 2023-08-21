import numpy as np
import pandas as pd
from darts import TimeSeries
import warnings
from sklearn.metrics import r2_score, mean_squared_error
from darts.metrics import mape, smape, mae
from PASTS.signal.test_statistiques import check_arguments


def root_mean_squared_error(y_true, y_pred, *args, **kwargs):
    return np.sqrt(mean_squared_error(y_true, y_pred, *args, **kwargs))


dict_metrics_sklearn = {'r2': r2_score,
                        'mse': mean_squared_error,
                        'rmse': root_mean_squared_error}

dict_metrics_darts = {'mape': mape,
                      'smape': smape,
                      'mae': mae}


class Metrics:

    def __init__(self, signal: "Signal", list_metrics: list[str]):
        self.signal = signal
        self.dict_metrics_sklearn = {key: dict_metrics_sklearn[key] for key in list_metrics
                                     if key in dict_metrics_sklearn.keys()}
        self.dict_metrics_darts = {key: dict_metrics_darts[key] for key in list_metrics
                                   if key in dict_metrics_darts.keys()}

    def scores_sklearn(self, model, axis):
        df_pred = self.signal.models[model]['predictions'].pd_dataframe()
        df_test = self.signal.test_data.copy()

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
                results.loc[col, metric] = self.dict_metrics_sklearn[metric](test, pred)

        return results

    def scores_darts(self, model):
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

    def compute_scores(self, model: str, axis):
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

        df_sk = self.scores_sklearn(model, axis)
        if axis == 0:
            warnings.warn('Only R2, MSE and RMSE can be computed for each date.')
            result = df_sk
        else:
            df_darts = self.scores_darts(model)
            result = pd.concat([df_sk, df_darts], axis=1)
        return result

    def scores_comparison(self, axis):
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





