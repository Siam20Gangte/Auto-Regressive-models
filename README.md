# Auto-Regressive-models
#  Time Series Forecasting - ARIMA, SARIMA and Auto-ARIMA


This is a repository dedicated to understanding time series data and forecasting future values. A time series is a sequence of data points that occur in successive order over some period of time. Time series data can be yearly, quarterly, monthly, weekly, daily, hourly, minutes, or even seconds in length, depending on the frequency. Some examples of time series data are air traffic, stock prices, weather, inbound calls in a call centre, and web traffic.

Forecasting is the next step in the process, and it involves predicting the series' future values. Time series forecasting entails developing models based on previous data and applying them to make observations and guide future strategic decisions. A key distinction in forecasting is that the future outcome is completely unknown at the time of the work and can only be anticipated by meticulous analysis and evidence-based priors.


The following installation and libraries should do the job

!pip install pmdarima
!pip install yfinance
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import yfinance as yf
from pmdarima.arima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
