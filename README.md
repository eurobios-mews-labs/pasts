# Toolbox for Time Series

This packages aims to structure the way time series analysis and forecasting is done. 

#### Purpose of the Package 
+ The purpose of the package is to
provide a collection of predicted models 
and analysis methods for time series in one unified library

#### Features 
+ Collection of analysis methods:
  - Scipy and statsmodel for testing 
  - times series processing
  - statistical testing (stationarity, check seasonality, ...  )
  - visualisation
+ Collection of forecasting models using Darts, which is itself an aggregator of 
   - scikit-learn
   - tenserflow
   - prophet
   - Auto regression models (ARIMA, SARIMA, ....)
   - etc.

#### Installation 
The package can be installed by :
```bash
python3 -m pip install git+https://gitlab.eurobios.com/escb/series-temporelles.git@series_biblio

```

### Usage and example

```python
import pandas as pd
import matplotlib.pyplot as plt

from darts.datasets import AirPassengersDataset
from darts.models import AutoARIMA, Prophet, ExponentialSmoothing

from seriestemporelles.signal.signal_analysis import SignalAnalysis
from seriestemporelles.signal.visualization import Visualisation

#---- Load data ---
series = AirPassengersDataset().load()
dt = pd.DataFrame(series.values())
dt.rename(columns={0: 'passengers'}, inplace=True)
dt.index = series.time_index

#---- Vizualise data ---
Visualisation(dt).plot_signal()
Visualisation(dt).plot_smoothing()
signal = SignalAnalysis(dt)
signal.profiling()
report = signal.apply_test('stationary', 'kpss')

#---- Machine Learning ---
timestamp = '1957-06-01'
signal.split_cv(timestamp=timestamp)

signal.apply_model(AutoARIMA())
signal.apply_model(Prophet())
signal.apply_model(ExponentialSmoothing())

# --- get some results ---
print(signal.scores)
exp_smoothing_pred = signal.results['ExponentialSmoothing']['predictions']

#---  Vizualise the predictions ---
signal.show_predictions()


```

### Author
