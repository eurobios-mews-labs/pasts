import pandas as pd
from pandas.plotting import autocorrelation_plot


class Visualisation:

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def plot_signal(self):
        self.data.plot(figsize=(20, 10), linewidth=5, fontsize=20)

    def plot_smoothing(self, resample_size: str = 'A', window_size: int = 12):
        """
         :param resample_size: calendar for resample, 'A' : year, 'D': Day, 'M': Month
         :type window_size: int : the size of window to rolling
         """
        resample_yr = self.data.resample(resample_size).mean()
        roll_yr = self.data.rolling(window_size).mean()
        ax = self.data.plot(alpha=0.5, style='-')  # store axis (ax) for latter plots
        resample_yr.plot(style=':', ax=ax)
        roll_yr.plot(style='--', ax=ax)
        ax.legend(['Resample at year frequency', 'Rolling average (smooth), window size=%s' % window_size])

    def acf_plot(self):
        autocorrelation_plot(self.data)
