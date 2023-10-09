import pytest
from darts.models import ExponentialSmoothing, AutoARIMA

from pasts.signal import Signal
from pasts.visualization import Visualization


def test_errors_visualization(get_univariate_data, get_multivariate_data):
    signal = Signal(get_univariate_data)
    signal_m = Signal(get_multivariate_data)
    with pytest.raises(Exception):
        Visualization(signal_m).acf_plot()
        Visualization(signal).show_predictions()


def test_plot_signal(get_univariate_data, get_multivariate_data):
    signal = Signal(get_univariate_data)
    signal_m = Signal(get_multivariate_data)
    Visualization(signal_m).plot_signal(display=False)
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)
    signal.apply_operations(['trend', 'seasonality'])
    Visualization(signal).plot_signal(display=False)


def test_acf_plot(get_univariate_data):
    signal = Signal(get_univariate_data)
    Visualization(signal).acf_plot()


def test_show_predictions(get_univariate_data):
    signal = Signal(get_univariate_data)
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)
    signal.apply_model(ExponentialSmoothing())
    Visualization(signal).show_predictions(display=False)


def test_show_forecast(get_univariate_data):
    signal = Signal(get_univariate_data)
    tstamp = '1958-12-01'
    signal.validation_split(tstamp)
    signal.apply_aggregated_model([AutoARIMA(), ExponentialSmoothing()])
    signal.forecast('AggregatedModel', 12)
    Visualization(signal).show_forecast(display=False)
