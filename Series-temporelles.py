#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:58:51 2023

@author: dcollot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.lines as mlines
import Objet_st as st
from datetime import timedelta

CSP = pd.read_csv('/home/dcollot/Bureau/Biogaran/CSP.csv', parse_dates=True, index_col='Période')
Previ = pd.read_csv('/home/dcollot/Bureau/Biogaran/prevision.csv', parse_dates=True, index_col='Période')
Det = pd.read_csv('/home/dcollot/Bureau/Biogaran/depos.csv',parse_dates=True,index_col='Période')

# k=Det['Code M'].unique()[15]
# Det_test=Det[Det['Code M']==k]
# Det2=Det_test[['CERP Rhin Rhône Med', 'IVRYLAB', 'CERP St Etienne','CERP Bretagne Nord','PHAR', 'EVOLUPHARM']]

index_r = int(len(CSP['Code M'])*random.random())
CSP_test = CSP[CSP['Code M'] == CSP['Code M'][index_r]]['Ventes CSP']
Previ_test = Previ[Previ['Code M'] == CSP['Code M'][index_r]]['Prévision M-1']
Det_test=Det[Det['Code M']==CSP['Code M'][index_r]]
Det2=Det_test[['CERP Rhin Rhône Med', 'IVRYLAB', 'CERP St Etienne','CERP Bretagne Nord','PHARMAR', 'EVOLUPHARM']]


Test=st.Inference(CSP_test,Previ_test)
Test.Test(1)
Test.fit

Test.Test(0)
Test.fit
            
SI = st.Signal(CSP_test, 10)
SI.diagnostic()
SI.get_trend('linear')
SI.remove('both')
SI.auto_correlation()

if SI.param_s > 1:
    SI.SARIMA_model()
else:
    SI.ARIMA_model()

plt.plot(CSP_test,color='red')
plt.plot(SI.prediction,color='blue',linestyle=':')
# Prediction interval, Naive forecast sigma(t)=sigma*sqrt(t); seasonal naive sigma(t)=sigma*sqrt(floor((t-1)/T)+1)
plt.fill_between(SI.prediction.index,
                 SI.prediction+1*SI.sigma*np.sqrt(np.floor((np.arange(len(SI.prediction))-1)/12)+1),
                 SI.prediction-1*SI.sigma*np.sqrt(np.floor((np.arange(len(SI.prediction))-1)/12)+1), color='blue', alpha=0.2)
plt.plot(Previ_test.loc[SI.data.index[-1]:SI.prediction.index[-1]],color='green',linestyle=':')    
plt.legend(['Data','Prediction','70% prediction interval','Biogaran Prediction'],bbox_to_anchor=(1.5, 1))
plt.title( CSP['Code M'][index_r])
plt.show()    

MSI=st.Multi_Signal(Det2,24)
MSI.VAR()
Colors=['blue','orange','green','gold','purple','brown','pink','grey','limegreen','cyan','magenta'] 
Lebals=['CERP Rhin Rhône Med', 'IVRYLAB', 'CERP St Etienne','CERP Bretagne Nord','PHAR', 'EVOLUPHARM']
i=0
blue_line=[]
for k in Det2.columns:
    plt.plot(pd.concat([Det2[k].iloc[24:],MSI.predict[k]]),color=Colors[i])
    blue_line.append(mlines.Line2D([],[], linestyle='-',color=Colors[i],marker='',label=Lebals[i]))
    plt.fill_between(MSI.predict[k].index,MSI.predict[k]+MSI.sigma[i],MSI.predict[k]-MSI.sigma[i],color=Colors[i],alpha=0.2)
    i=i+1
GH=list(pd.concat([Det2[k].iloc[24:],MSI.predict[k]]).index)
GH=[i.strftime('%Y-%m') for i in GH]
plt.axvline(Det2.iloc[-1].name,linestyle=':')
plt.legend(handles=blue_line,bbox_to_anchor=(1,1))
plt.title(Det['Code M'].unique()[15])
plt.show()   
