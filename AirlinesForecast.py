# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 18:13:37 2020

@author: sowmi
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from pandas import Series
from pandas.plotting import register_matplotlib_converters

from    pandas             import   read_csv, Grouper, DataFrame, concat
import  matplotlib.pyplot  as       plt
import  statsmodels.api          as       sm
from   sklearn.metrics      import  mean_squared_error
from   statsmodels.tsa.holtwinters     import  SimpleExpSmoothing,Holt, ExponentialSmoothing
import statsmodels.tsa.holtwinters     as      ets
import statsmodels.tools.eval_measures as      fa
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import sys
import warnings
import itertools
warnings.filterwarnings("ignore")
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
%matplotlib inline
register_matplotlib_converters()
air=pd.read_csv("C:\\Users\\tussh\\Documents\\Project\\airlines.csv")
air.tail()
air.head()
air.columns=["Month","Passangers"]
# Convert Month into Datetime
air['Month']=pd.to_datetime(air['Month'])

air.head()
air.set_index('Month',inplace=True)
air.describe()
air.isnull().sum
#Visualize the Data
air.plot()
### Testing For Stationarity
#Rolling statistics
rolmean=air.rolling(window=12).mean()
rolsd=air.rolling(window=12).std()
print(rolmean,rolsd)

#plotting rolling statistics
orig=plt.plot(air,color="blue",Label="Original")
mean=plt.plot(rolmean,color="red",Label="Rolling Mean")
std=plt.plot(rolsd,color="green",Label="Rolling Std Dev")
plt.legend(loc="best")
plt.title("rolling Mean & Std Dev")
plt.show(block=False)

##ADCF Test
from statsmodels.tsa.stattools import adfuller
test_result=adfuller(air['Passangers'])
#Ho: It is non stationary
#H1: It is stationary

print("The Resultsof DCF method")
dftest=adfuller(air["Passangers"],autolag="AIC")
dfoutput=pd.Series(dftest[0:4],index=["Test Statistics","p-value","Num of lags","num of Observations"])
for key,value in dftest[4].items():
    dfoutput["Critiak Valueat (%s)"%key]=value
    
print(dfoutput)    

########### or ############
#def adfuller_test(Passangers):
#    result=adfuller(Passangers)
#    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
#    for value,label in zip(result,labels):
#        print(label+' : '+str(value) )
#    if result[1] <= 0.05:
#        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
#    else:
#        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
        
#adfuller_test(air['Passangers'])        



#1st differenceing since data is not stationary 
air['Passangers First Difference'] = air['Passangers'] - air['Passangers'].shift(1)
air['Passangers'].shift(1)
#seasonality difference
#air['Seasonal First Difference']=air['Sales']-air['Sales'].shift(12)
#air.head(14)
## Again test dickey fuller test
adfuller_test(air['Passangers First Difference'].dropna())
air['Passangers First Difference'].plot()

# 2nd Difference test
air['Passangers Second Difference'] = air['Passangers'] - air['Passangers'].shift(2)
air['Passangers'].shift(2)
dftest=adfuller(air['Passangers Second Difference'].dropna(),autolag='AIC')
print(dftest)
air['Passangers Second Difference'].plot()



#Auto Regressive Model
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(air['Passangers'])
plt.show()

#finding p and q values
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
fig=plot_acf(air["Passangers Second Difference"].dropna())
fig=plot_pacf(air["Passangers Second Difference"].dropna())

#####################Seasonal First Difference plot
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(air['Seasonal First Difference'].iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(air['Seasonal First Difference'].iloc[13:],lags=40,ax=ax2)
###########################

# For non-seasonal data
#p=1, d=1, q=0 or 1
from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(air['Passangers'],order=(1,1,1))
model_fit=model.fit()
model_fit.summary()


air['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
air[['Passangers','forecast']].plot(figsize=(12,8))


#### Seasonal ARIMA TEST
import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(air['Passangers'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()

air['forecast']=results.predict(start=90,end=103,dynamic=True)
air[['Passangers','forecast']].plot(figsize=(12,8))

from pandas.tseries.offsets import DateOffset
future_dates=[air.index[-1]+ DateOffset(months=x)for x in range(0,24)]

future_datest_df=pd.DataFrame(index=future_dates[1:],columns=air.columns)
future_datest_df.tail()


future_df=pd.concat([air,future_datest_df])
future_df['forecast'] = results.predict(start = 104, end = 120, dynamic= True)  
future_df[['Passangers', 'forecast']].plot(figsize=(12, 8))
