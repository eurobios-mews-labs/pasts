#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:48:34 2023

@author: dcollot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

class Inference:
    def __init__(self,data,predict):
        self.data=data
        self.predict=predict
        self.time_step = (self.data.index[1]-self.data.index[0]).days
        if self.time_step == 29 or self.time_step == 30:
            self.time_step = 31
        self.completude = ~((self.data.index[1:]-self.data.index[:len(self.data)-1]).days > self.time_step).any()
        self.nb_year = ((max(self.data.index)-min(self.data.index)).days++30*(self.time_step == 31))/365
        self.date1=self.data.index[0]
        self.dateM=self.data.index[-1]
        
    def _generate_dataframe(self,k):
        from datetime import date        
        self.date2=date.fromisoformat('20'+str(k)+'-12-01')
        self.date3=date.fromisoformat('20'+str(k+1)+'-01-01')
        self.date4=date.fromisoformat('20'+str(k+1)+'-12-01')
        self.training=self.data.loc[self.date1:self.date2,]
        self.test=self.data.loc[self.date3:self.date4,]
        self.predict_test=self.predict.loc[self.date3:self.date4,]
        
    def likelihood(self,mus,sigma):
        from scipy.stats import norm
        from numpy import prod
        liste=norm.pdf(self.test,mus,sigma)
        self.lh=prod(liste)
       
    def Test(self):
        #from datetime import date
        import numpy as np
        d1=int(self.date1.strftime('%Y'))-2000
        d2=int(self.dateM.strftime('%Y'))-2000
        Results=[]
        for k in np.arange(d1+1,d2):
            
            self._generate_dataframe(k)           
            SI=Signal(self.training,12)
            
            SI.diagnostic()
            SI.auto_correlation()
            if SI.param_s > 1:
                SI.SARIMA_model()
            else:
                SI.ARIMA_model()
            
            Results.append([sum(abs(SI.prediction-self.test).dropna()),100*(sum(abs(SI.prediction-self.test).dropna())/sum(self.test)),sum(abs(self.predict_test-self.test).dropna())])
        
        Results=pd.DataFrame(Results,columns=['model','% erreur model','predict'], index=2000+np.arange(d1+2,d2+1))
        Results['Model/predict']=Results['model']/Results['predict']
        self.fit=Results

class Signal:
    
    def __init__(self, data, forecast):
        self.data = data
        self.forecast = forecast+len(data)
        self.time_step = (self.data.index[1]-self.data.index[0]).days
        if self.time_step == 29 or self.time_step == 30:
            self.time_step = 31
        self.completude = ~((self.data.index[1:]-self.data.index[:len(self.data)-1]).days > self.time_step).any()
        self.nb_year = ((max(self.data.index)-min(self.data.index)).days++30*(self.time_step == 31))/365
        self.trend_linear_removed = 0
        self.trend_period_removed = 0
        self.got_trend = 0
        self.param_p = 1
        self.param_q = 1
        self.param_d = 0
        self.param_s = 1
        if 1-self.completude:
            print('The time step is not regular. The accuracy of the ARIMA method can be lower.')
        else:
            if self.time_step == 31:
                self.data.index.freq = 'MS'
            elif self.time_step == 1:
                self.data.index.freq = 'D'
            elif self.time_step == 365:
                self.data.index.freq = 'A'
    
    def diagnostic(self):
        from statsmodels.tsa.stattools import adfuller
        p_val_0 = adfuller(self.data)[1]
        print('ADF test: p_val='+str(p_val_0))
        if p_val_0 < 0.05:
            print(' Data are stationnary')
        else:
            print('Data are non-stationnary')    
            p_val_1 = adfuller((self.data-self.data.shift(1)).dropna())[1]
            print('ADF test after one month shift: p_val='+str(p_val_1))
            if p_val_1 < 0.05:
                self.param_d = 1
            
            p_val_12 = adfuller((self.data-self.data.shift(12)).dropna())[1]
            print('ADF test after 12 months shift: p_val='+str(p_val_12))
            if p_val_12 < 0.05:
                self.param_s = 12

    def auto_correlation(self):
        from statsmodels.tsa.stattools import pacf, acf
        from scipy.stats import chi2
        
        PACF = pacf(self.data)
        ACF_val = acf(self.data, qstat=True)[0]
        # ACF_pval = acf(self.data, qstat=True)[2]
        
        Reject_value = [0]
        n = len(self.data)
        G = np.sqrt(chi2.ppf(0.95, 1)/n/(n+2)*(n-1))
        for k in np.arange(2, len(ACF_val)):
            G = G*(n-k)/(n-k-1)+ACF_val[k-1]*ACF_val[k-1]/(n-k-1)
            Reject_value.append(G)
            
        i = 0
        while abs(ACF_val[i]) > Reject_value[i] and i < (len(ACF_val)-1):
            i = i+1
        
        j = 0
        while abs(PACF[j]) > Reject_value[1] and j < (len(PACF)-1):
            j = j+1
           
        print('AC: significative until order: '+str(i-1))
        print('Partial AC: significative until order: '+str(j-1))
        
        if i == 1 and j == 1:
            print('The autocorrelation is low, still worth trying p=1 and q=1.')
            print('These values can be changed using \'.parameter(p,q)\'.')
            self.param_p = 1
            self.param_q = 1
        else:
            print('It suggests using p='+str(j-1)+' and q='+str(i-1))
            print('These values can be changed using \'.parameter(p,q)\'.')
            self.param_p = j-1
            self.param_q = i-1
            
    def ARIMA_model(self):
        from statsmodels.tsa.arima.model import ARIMA
        
        model = ARIMA(self.data, order=(self.param_p, self.param_d, self.param_q))
        results = model.fit()
        self.prediction = results.predict(start=len(self.data)-2, end=self.forecast)
        self.sigma = pow(results.params['sigma2'], 0.5)
        
    def SARIMA_model(self):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        model = SARIMAX(self.data, order=(self.param_p, self.param_d, self.param_q), seasonal_order=(self.param_p, self.param_d, self.param_q, self.param_s))
        results = model.fit()
        self.prediction = results.predict(start=len(self.data)-2, end=self.forecast)
        self.sigma = pow(results.params['sigma2'], 0.5)
    
    def get_trend(self, method='rolling mean'):
        from statsmodels.tsa.seasonal import seasonal_decompose
        from scipy import stats
        from statsmodels.tsa.tsatools import detrend

        self.slope = stats.linregress(range(len(self.data)), self.data).slope
        self.intercept = stats.linregress(range(len(self.data)), self.data).intercept
        self.p_valeur = stats.linregress(range(len(self.data)), self.data).pvalue
        print('For your information, the p_val of the linear regression is '+str(round(self.p_valeur, 4)))
        
        if method == 'rolling mean':
            self.trend_linear = seasonal_decompose(self.data, model='additive').trend
        elif method == 'linear':
            self.trend_linear = self.data-detrend(self.data)
        else: 
            print('This method does not exist, please use \'rolling mean\' or \'linear\'.')
        self.trend_period = seasonal_decompose(self.data, model='additive').seasonal
        self.got_trend = 1
    
    def remove(self, method):
        
        if 1-self.got_trend:
            from statsmodels.tsa.seasonal import seasonal_decompose
            self.trend_linear = seasonal_decompose(self.data, model='additive').trend
            self.trend_period = seasonal_decompose(self.data, model='additive').seasonal
            self.got_trend = 1
        
        if method == 'linear' or method == 'both':
            if 1-self.trend_linear_removed:
                self.data = self.data-self.trend_linear
                self.trend_linear_removed = 1
            else:
                print("the linear trend has already been removed.")
                
        if method == 'periodic' or method == 'both':
            if 1-self.trend_period_removed:
                self.data = self.data-self.trend_period
                self.trend_period_removed = 1
            else:
                print("the periodic trend has already been removed.")

        if not (method in ['linear', 'periodic', 'both']):
            print("Method unknown, please use 'linear', 'periodic' or 'both'.")


# ########################### Test ###########################

CSP = pd.read_csv('/home/dcollot/Bureau/Biogaran/CSP.csv', parse_dates=True, index_col='Période')
Previ = pd.read_csv('/home/dcollot/Bureau/Biogaran/prevision.csv', parse_dates=True, index_col='Période')

index_r = int(len(CSP['Code M'])*random.random())
CSP_test = CSP[CSP['Code M'] == CSP['Code M'][index_r]]['Ventes CSP']
Previ_test = Previ[Previ['Code M'] == CSP['Code M'][index_r]]['Prévision M-1']

Test=Inference(CSP_test,Previ_test)
Test.Test()
            
SI = Signal(CSP_test, 10)
SI.diagnostic()
SI.get_trend()
SI.auto_correlation()

# SI.parameter(1,0,1,12)
if SI.param_s > 1:
    SI.SARIMA_model()
else:
    SI.ARIMA_model()

plt.plot(SI.data,color='red')
plt.plot(SI.prediction,color='blue',linestyle=':')
# Prediction interval, Naive forecast sigma(t)=sigma*sqrt(t); seasonal naive sigma(t)=sigma*sqrt(floor((t-1)/T)+1)
plt.fill_between(SI.prediction.index,
                 SI.prediction+1*SI.sigma*np.sqrt(np.floor((np.arange(len(SI.prediction))-1)/12)+1),
                 SI.prediction-1*SI.sigma*np.sqrt(np.floor((np.arange(len(SI.prediction))-1)/12)+1), color='blue', alpha=0.2)
plt.plot(Previ_test.loc[SI.data.index[-1]:SI.prediction.index[-1]],color='green',linestyle=':')    
plt.legend(['Data','Prediction','70% prediction interval','Biogaran Prediction'],bbox_to_anchor=(1.5, 1))
plt.title( CSP['Code M'][index_r])
plt.show()    

Test.fit   

# ########################## Future Objet Operation ##############################


class Operation:
    
    def __init__(self):
        self.op = 0
        
    def removing(self):
        ...
        
    def adding(self):
        ...
    