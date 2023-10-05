import pandas as pd
from darts.datasets import AirPassengersDataset, AustralianTourismDataset
from pytest import fixture
from pasts.signal import Signal


@fixture(scope="module")
def get_univariate_data():
    series = AirPassengersDataset().load()
    dt = pd.DataFrame(series.values())
    dt.rename(columns={0: 'passengers'}, inplace=True)
    dt.index = series.time_index
    return dt


@fixture(scope='module')
def get_multivariate_data():
    series = AustralianTourismDataset().load()[['Hol', 'VFR', 'Oth']]
    df = pd.DataFrame(series.values())
    df.rename(columns={0: 'Hol', 1: 'VFR', 2: 'Oth'}, inplace=True)
    df.index = pd.date_range(start='2020-01-01', freq='MS', periods=len(df))
    return df
