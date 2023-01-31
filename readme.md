# Toolbox for Time Series

## Install notice

## Approach

This package contains two main modules :

1. A Signal mmodule, that enable user to perform parameters estimation for a SARIMA model, and forecast using this model. 

2. A Test module, that enable the user to perform validation of the model. 
    
## Basic usage

### Data set

The data set provided must be time series, i.e. a dataframe with datetime as index.

The following instruction open a dataset (CSP) and a Prediction set (Previ). One specific individual is randomly selected form the 1173 available.  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

CSP = pd.read_csv('/home/dcollot/Bureau/Biogaran/CSP.csv', parse_dates=True, index_col='Période')
Previ = pd.read_csv('/home/dcollot/Bureau/Biogaran/prevision.csv', parse_dates=True, index_col='Période')

index_r = int(len(CSP['Code M'])*random.random())
CSP_test = CSP[CSP['Code M'] == CSP['Code M'][index_r]]['Ventes CSP']
Previ_test = Previ[Previ['Code M'] == CSP['Code M'][index_r]]['Prévision M-1']
```

The Signal module preforms the estimation of paramters "phi" using the SARIMAX function of the package statsmodels. The estimation of parameters (p,d,q,s) is automatically made, but can be modified by the users.
The .diagnostic method allows the estimation of d and s, while .auto_correlation allows the estimation of p and q. 
Once this two steps are made, the .SARIMA_model can be performed (or ARIMA_model if s==1, can be automatized).

```python
SI = Signal(CSP_test, 72)
SI.diagnostic()
SI.get_trend()
SI.auto_correlation()

# SI.parameter(1,0,1,12)
if SI.param_s > 1:
    SI.SARIMA_model()
else:
    SI.ARIMA_model()

plt.plot(SI.data)
plt.plot(SI.prediction)
# Prediction interval, Naive forecast sigma(t)=sigma*sqrt(t); seasonal naive sigma(t)=sigma*sqrt(floor((t-1)/T)+1)
plt.fill_between(SI.prediction.index,
                 SI.prediction+1.96*SI.sigma*np.sqrt(np.floor((np.arange(len(SI.prediction))-1)/12)+1),
                 SI.prediction-1.96*SI.sigma*np.sqrt(np.floor((np.arange(len(SI.prediction))-1)/12)+1), alpha=0.5)
plt.plot(Previ_test)    
plt.show()    
```

The model then return .prediction and .sigma that enable the plot of the results. The estimation of prediction intervals is not so clear.

### Trends

The Signal module also compute the trends of the time series. The ARIMA model already take them into account,  it is therefore not necessary to remove them. 
The methods .get-trend() and _remove() can be used to detrend a signla if needed. 


### Test of model performance

The Test module performs the comparison of the data sets and the estimations made by the model. The calibration is made from year 1 to N-1, and the prediction are made using year N.
It is possible to give reference estimation adn compare the performance of the model to the reference (could be modified to allow empy dataframe of reference).


```python
Test=Inférence(CSP_test,Previ_test)
Test.Test()
Test.fit   
```
The module returns .fit that gives the sum of the absolute difference between the prediction and the data for each year N.
 

## Authors

* Dorian COLLOT : dorian.collot@eurobios.com

<img src="https://www.mews-partners.com/wp-content/uploads/2021/09/Eurobios-Mews-Labs-logo-768x274.png.webp" width="200"/>
