from seriesTemporelles.src.signal import Signal
from darts.datasets import AirPassengersDataset
import pandas as pd
from darts.models import ExponentialSmoothing
from darts import TimeSeries

series = AirPassengersDataset().load()
dt = pd.DataFrame(series.values())
dt.rename(columns={0: 'passengers'})
dt.index = series.time_index
signal = Signal(dt)
signal.profiling()
signal.plot_signal()
signal.plot_smoothing()
signal.adf_test(0.05)
