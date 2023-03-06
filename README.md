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
python -m pip install https://gitlab.eurobios.com/escb/series-temporelles.git

```

### Usage and example
```python
from darts.datasets import AirPassengersDataset
import pandas as pd
from seriestemporelles import UniSignal
from statsmodels.tsa.stattools import acf, adfuller, grangercausalitytests, kpss, pacf

series = AirPassengersDataset().load()
dt = pd.DataFrame(series.values())
dt.rename(columns={0: 'passengers'})
dt.index = series.time_index
signal = UniSignal(dt)
signal.profiling()
signal.plot_signal()
signal.plot_smoothing()
report = signal.is_stationary(test_stat=kpss, regression='c', nlags="auto")
```

### Author