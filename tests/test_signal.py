import pytest
from darts.models import ExponentialSmoothing, AutoARIMA
from darts.utils.utils import ModelMode, SeasonalityMode
import math

from pasts.signal import Signal


def test_validation_split(get_univariate_data):
    signal = Signal(get_univariate_data)
    tstamp = '1949-01-01'
    with pytest.raises(ValueError):
        signal.validation_split(tstamp)
    tstamp2 = '1958-12-01'
    signal.validation_split(tstamp2, n_splits_cv=5)
    assert signal.test_data.shape[0] == 24
    assert signal.train_data.shape[0] == 120


def test_apply_model(get_univariate_data):
    signal = Signal(get_univariate_data)
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)
    signal.apply_model(ExponentialSmoothing())
    signal.apply_model(AutoARIMA())
    assert len(signal.models) == 2
    for model in signal.models.keys():
        assert len(signal.models[model]) == 4
        assert len(signal.models[model]['predictions']) == 24
        assert signal.models[model]['best_parameters'] == "default"
    assert isinstance(signal.models['ExponentialSmoothing']['estimator'], ExponentialSmoothing)
    assert isinstance(signal.models['AutoARIMA']['estimator'], AutoARIMA)


def test_apply_model_grid(get_univariate_data):
    signal = Signal(get_univariate_data)
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)
    with pytest.raises(Exception):
        signal.apply_model(model=ExponentialSmoothing(), gridsearch=True)
    param_grid = {'trend': [ModelMode.ADDITIVE, ModelMode.MULTIPLICATIVE, ModelMode.NONE],
                  'seasonal': [SeasonalityMode.ADDITIVE, SeasonalityMode.MULTIPLICATIVE, SeasonalityMode.NONE],
                  }
    signal.apply_model(ExponentialSmoothing(), gridsearch=True, parameters=param_grid)
    assert len(signal.models) == 1
    assert len(signal.models['ExponentialSmoothing']) == 4
    assert len(signal.models['ExponentialSmoothing']['predictions']) == 24
    assert signal.models['ExponentialSmoothing']['best_parameters'] != "default"
    assert len(signal.models['ExponentialSmoothing']['best_parameters']) == 2
    assert isinstance(signal.models['ExponentialSmoothing']['estimator'], ExponentialSmoothing)


def test_aggregated_model(get_univariate_data):
    signal = Signal(get_univariate_data)
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)
    signal.apply_aggregated_model([AutoARIMA(), ExponentialSmoothing()])
    assert len(signal.models) == 3
    assert len(signal.models['AggregatedModel']['predictions']) == 24
    assert signal.models['AggregatedModel']['weights'].shape == (1, 2)
    assert len(signal.models['AggregatedModel']['models']) == 2
    assert round(signal.models['AggregatedModel']['weights'].sum(axis=1)[0], 0) == 1


def test_scores_unit(get_univariate_data):
    signal = Signal(get_univariate_data)
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)
    signal.apply_aggregated_model([AutoARIMA(), ExponentialSmoothing()])

    signal.compute_scores(['r2', 'mse', 'mape'])
    assert signal.models['ExponentialSmoothing']['scores']['unit_wise'].shape[1] == 3
    assert signal.models['AutoARIMA']['scores']['unit_wise'].shape[1] == 3
    assert signal.models['AggregatedModel']['scores']['unit_wise'].shape[1] == 3
    assert len(signal.performance_models) == 1
    assert len(signal.performance_models['unit_wise']) == 3

    signal.compute_scores()
    for model in signal.models.keys():
        assert not signal.models[model]['scores']['time_wise']
        assert signal.models[model]['scores']['unit_wise'].shape[1] == 6
        assert signal.models[model]['scores']['unit_wise'].loc['passengers', 'rmse']**2 - 0.1 < \
               signal.models[model]['scores']['unit_wise'].loc['passengers', 'mse'] <\
               signal.models[model]['scores']['unit_wise'].loc['passengers', 'rmse']**2 + 0.1
    assert len(signal.performance_models) == 1
    assert len(signal.performance_models['unit_wise']) == 6
    for metric in signal.performance_models['unit_wise'].keys():
        assert signal.performance_models['unit_wise'][metric].shape == (1, 3)


def test_scores_time(get_univariate_data):
    signal = Signal(get_univariate_data)
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)
    signal.apply_aggregated_model([AutoARIMA(), ExponentialSmoothing()])

    signal.compute_scores(['r2', 'mse', 'mape'], axis=0)
    for model in signal.models.keys():
        assert signal.models[model]['scores']['time_wise'].shape[1] == 1
    assert len(signal.performance_models) == 1
    assert len(signal.performance_models['time_wise']) == 1

    signal.compute_scores(axis=0)
    for model in signal.models.keys():
        assert not signal.models[model]['scores']['unit_wise']
        assert signal.models[model]['scores']['time_wise'].shape[1] == 2
    assert len(signal.performance_models) == 1
    assert len(signal.performance_models['time_wise']) == 2
    for metric in signal.performance_models['time_wise'].keys():
        assert signal.performance_models['time_wise'][metric].shape == (24, 3)


def test_forecast(get_univariate_data):
    signal = Signal(get_univariate_data)
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)
    with pytest.raises(Exception):
        signal.forecast('AggregatedModel', 12)
    signal.apply_aggregated_model([AutoARIMA(), ExponentialSmoothing()])
    signal.forecast('AggregatedModel', 12)
    for model in signal.models['AggregatedModel']['models'].keys():
        assert len(signal.models[model]) == 6
        assert signal.models[model]['forecast'].n_timesteps == 12
    assert len(signal.models['AggregatedModel']) == 5
    assert signal.models['AggregatedModel']['forecast'].n_timesteps == 12
    assert signal.models['AggregatedModel']['forecast'].time_index[0] > signal.data.index[-1]


def test_properties(get_univariate_data, get_multivariate_data):
    signal = Signal(get_univariate_data)
    assert signal.properties['shape'] == (144, 1)
    assert signal.properties['is_univariate'] == True
    assert len(signal.properties) == 5
    signal_m = Signal(get_multivariate_data)
    assert signal_m.properties['is_univariate'] == False


def test_adfuller(get_univariate_data, get_multivariate_data):
    signal = Signal(get_univariate_data)
    signal.apply_stat_test('stationary')
    assert 'stationary: adfuller' in signal.tests_stat.keys()
    assert not signal.tests_stat['stationary: adfuller'][0]
    assert signal.tests_stat['stationary: adfuller'][1] > 0.95
    signal_m = Signal(get_multivariate_data)
    with pytest.raises(TypeError):
        signal_m.apply_stat_test('stationary')


def test_kpss(get_univariate_data, get_multivariate_data):
    signal = Signal(get_univariate_data)
    signal.apply_stat_test('stationary', 'kpss')
    assert 'stationary: kpss' in signal.tests_stat.keys()
    assert not signal.tests_stat['stationary: kpss'][0]
    assert signal.tests_stat['stationary: kpss'][1] < 0.05
    signal_m = Signal(get_multivariate_data)
    with pytest.raises(TypeError):
        signal_m.apply_stat_test('stationary', 'kpss')


def test_seasonality(get_univariate_data, get_multivariate_data):
    signal = Signal(get_univariate_data)
    signal.apply_stat_test('seasonality')
    assert 'seasonality: check_seasonality' in signal.tests_stat.keys()
    assert signal.tests_stat['seasonality: check_seasonality'][1] == (True, 12)
    signal_m = Signal(get_multivariate_data)
    with pytest.raises(TypeError):
        signal_m.apply_stat_test('seasonality')


def test_causality(get_univariate_data, get_multivariate_data):
    signal = Signal(get_univariate_data)
    with pytest.raises(TypeError):
        signal.apply_stat_test('causality')
    signal_m = Signal(get_multivariate_data)
    signal_m.apply_stat_test('causality')
    assert len(signal_m.tests_stat['causality: grangercausalitytests']) == math.perm(3, 2)
    for t in signal_m.tests_stat['causality: grangercausalitytests'].keys():
        assert ((signal_m.tests_stat['causality: grangercausalitytests'][t][0] == True) and
                (signal_m.tests_stat['causality: grangercausalitytests'][t][1] < 0.05)) or \
               ((signal_m.tests_stat['causality: grangercausalitytests'][t][0] == False) and
                (signal_m.tests_stat['causality: grangercausalitytests'][t][1] > 0.05))


def test_operations(get_univariate_data):
    signal = Signal(get_univariate_data)
    with pytest.raises(Exception):
        signal.apply_operations(['trend', 'seasonality'])
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)
    with pytest.raises(Exception):
        signal.apply_operations(['outliers'])
    signal.apply_operations(['trend', 'seasonality'])
    assert signal.rest_data.shape == signal.data.shape
    assert signal.rest_train_data.shape == signal.train_data.shape

