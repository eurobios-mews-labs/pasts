# Toolbox for Time Series

This package aims to structure the way time series analysis and forecasting is done. 

#### Purpose of the Package 
+ The purpose of the package is to
provide a collection of forecasting models 
and analysis methods for time series in one unified library.

#### Features 
+ Collection of analysis methods:
  - Scipy and statsmodel for testing 
  - Time series processing
  - Statistical testing (stationarity, check seasonality, ...  )
  - Visualization
+ Collection of forecasting models using Darts, which is itself an aggregator of 
   - scikit-learn
   - tenserflow
   - prophet
   - Auto regression models (ARIMA, SARIMA, ....)
   - etc.

#### Installation 
The package can be installed by :
```bash
python3 -m pip install git+https://gitlab.eurobios.com/escb/series-temporelles.git@aggregated_model

```

### Usage and example

```python
import pandas as pd

from darts.datasets import AirPassengersDataset, AustralianTourismDataset
from darts.models import AutoARIMA, Prophet, ExponentialSmoothing, XGBModel, VARIMA
from darts.utils.utils import ModelMode, SeasonalityMode

from pasts.signal import Signal
from pasts.visualization import Visualization

# ---- Load data ----
series = AirPassengersDataset().load()
dt = pd.DataFrame(series.values())
dt.rename(columns={0: 'passengers'}, inplace=True)
dt.index = series.time_index

# --- Visualize data ---
signal = Signal(dt)
print(signal.properties)
Visualization(signal).plot_signal()
Visualization(signal).acf_plot()
signal.apply_stat_test('stationary')
signal.apply_stat_test('stationary', 'kpss')
signal.apply_stat_test('seasonality')

# --- Machine Learning ---
timestamp = '1958-12-01'
signal.validation_split(timestamp=timestamp)
signal.apply_operations(['trend', 'seasonality'])
Visualization(signal).plot_signal()

signal.apply_model(ExponentialSmoothing())

signal.apply_model(AutoARIMA())
signal.apply_model(Prophet())

param_grid = {'trend': [ModelMode.ADDITIVE, ModelMode.MULTIPLICATIVE, ModelMode.NONE],
              'seasonal': [SeasonalityMode.ADDITIVE, SeasonalityMode.MULTIPLICATIVE, SeasonalityMode.NONE],
              }
signal.apply_model(ExponentialSmoothing(), gridsearch=True, parameters=param_grid)

# --- Compute scores ---
signal.compute_scores()
signal.compute_scores(axis=0)

# ---  Visualize predictions ---
Visualization(signal).show_predictions()

# --- Aggregated Model ---
signal.apply_aggregated_model([ExponentialSmoothing(), Prophet()])
signal.compute_scores(axis=1)
Visualization(signal).show_predictions()

# --- Forecast ---
signal.forecast("Prophet", 100)
signal.forecast("AggregatedModel", 100)
signal.forecast("AutoARIMA", 100)
signal.forecast("ExponentialSmoothing", 100)

# --- Visualize forecasts ---
Visualization(signal).show_forecast()
```

### Author
