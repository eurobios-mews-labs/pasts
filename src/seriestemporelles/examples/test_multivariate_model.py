import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from darts.datasets import AustralianTourismDataset
from darts.models import VARIMA, XGBModel
from seriestemporelles.signal.signal_analysis import MultiVariateSignalAnalysis
from seriestemporelles.signal.visualization import Visualisation

if __name__ == '__main__':
    # ---- Load data ---
    series = AustralianTourismDataset().load()[['Hol', 'VFR', 'Oth']]
    df = pd.DataFrame(series.values())
    df.rename(columns={0: 'Hol', 1: 'VFR', 2: 'Oth'}, inplace=True)
    df.index = series.time_index

    # ---- Vizualise data ---
    Visualisation(df).plot_signal()

    # ---- Machine Learning ---
    MultiVariateSignal = MultiVariateSignalAnalysis(df)

    timestamp = 30
    MultiVariateSignal.split_cv(timestamp=timestamp)

    MultiVariateSignal.apply_model(VARIMA())
    MultiVariateSignal.apply_model(XGBModel(lags=[-1, -2, -3]))

    # ---  Vizualise the predictions ---
    MultiVariateSignal.show_predictions()
