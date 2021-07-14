import pandas as pd
import numpy as np
import warnings
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from datetime import datetime, timedelta
import scipy.stats as st
import statsmodels.api as sm
import math
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
from scipy import poly1d
warnings.simplefilter(action='ignore',  category=Warning)
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from pandas.tseries.offsets import BDay
from plotly.subplots import make_subplots
matplotlib.rcParams['figure.figsize'] = (25.0, 15.0)
matplotlib.style.use('ggplot')
pd.set_option('display.float_format', lambda x: '%.4f' % x)
import plotly.io as pio
from numpy import median, mean

pio.templates.default = "plotly_white"

def clean_contract_data(train, year, month, offset):
    
    Month_dict = {
        'Jan': {'Front':[1,12,10,9,8,7,5,3,1],'Back':[11,9,8,7,6,4,2,12,11]},
        'Mar': {'Front':[3,1,12,10,9,8,7,5,3],'Back':[12,11,9,8,7,6,4,2,12]},
        'May': {'Front':[5,3,1,12,10,9,8,7,5],'Back':[2,12,11,9,8,7,6,4,2]},
        'Jul': {'Front':[7,5,3,1,12,10,9,8,7],'Back':[4,2,12,11,9,8,7,6,4]},
        'Aug': {'Front':[8,7,5,3,1,12,10,9,8],'Back':[6,4,2,12,11,9,8,7,6]},
        'Sep': {'Front':[9,8,7,5,3,1,12,10,9],'Back':[7,6,4,2,12,11,9,8,7]},
        'Oct': {'Front':[10,9,8,7,5,3,1,12,10],'Back':[8,7,6,4,2,12,11,9,8]},
        'Dec': {'Front':[12,10,9,8,7,5,3,1,12],'Back':[9,8,7,6,4,2,12,11,9]},
                }
    front_year_condition_check = Month_dict[month]['Front'].index(1) < Month_dict[month]['Front'].index(Month_dict[month]['Front'][offset])
    back_year_condition_check = Month_dict[month]['Back'].index(2) < Month_dict[month]['Back'].index(Month_dict[month]['Back'][offset])

    if front_year_condition_check == False: 
        year_actual_front = 2000 + year

    elif front_year_condition_check == True: 
        year_actual_front = 2000 + year - 1

    if back_year_condition_check==False:
        year_actual_back = 2000+year

    elif back_year_condition_check==True:
        year_actual_back = 2000 + year - 1

    back_month  = Month_dict[month]['Back'][offset]

    front_month = Month_dict[month]['Front'][offset]

    if back_month <= 9:
        back_separator = '-0'
    elif back_month >= 10:
        back_separator = '-'

    if front_month <= 9:
        front_separator = '-0'
    elif front_month >= 10:
        front_separator = '-'

    start_day = 10
    end_day = 1

    if start_day <=9:
        start_day_separator = '-0'
    elif start_day>=10:
        start_day_separator = '-'

    if end_day <=9:
        end_day_separator = '-0'
    elif end_day>=10:
        end_day_separator = '-'

    start_date = str(year_actual_back) + back_separator +str(back_month) + start_day_separator + str(start_day)
    end_date = str(year_actual_front) + front_separator +str(front_month) + end_day_separator + str(end_day)

    start_date = pd.to_datetime(start_date, infer_datetime_format=True)
    end_date = pd.to_datetime(end_date, infer_datetime_format=True)
    data_mask = (train.index < end_date) & (train.index >= start_date)

    train = train.loc[data_mask]

    return train

def clean_backtest_data(train, tradeLog, year, month, offset):
    
    Month_dict = {
        'Jan': {'Front':[1,12,10,9,8,7,5,3,1],'Back':[11,9,8,7,6,4,2,12,11]},
        'Mar': {'Front':[3,1,12,10,9,8,7,5,3],'Back':[12,11,9,8,7,6,4,2,12]},
        'May': {'Front':[5,3,1,12,10,9,8,7,5],'Back':[2,12,11,9,8,7,6,4,2]},
        'Jul': {'Front':[7,5,3,1,12,10,9,8,7],'Back':[4,2,12,11,9,8,7,6,4]},
        'Aug': {'Front':[8,7,5,3,1,12,10,9,8],'Back':[6,4,2,12,11,9,8,7,6]},
        'Sep': {'Front':[9,8,7,5,3,1,12,10,9],'Back':[7,6,4,2,12,11,9,8,7]},
        'Oct': {'Front':[10,9,8,7,5,3,1,12,10],'Back':[8,7,6,4,2,12,11,9,8]},
        'Dec': {'Front':[12,10,9,8,7,5,3,1,12],'Back':[9,8,7,6,4,2,12,11,9]},
                }
    front_year_condition_check = Month_dict[month]['Front'].index(1) < Month_dict[month]['Front'].index(Month_dict[month]['Front'][offset])
    back_year_condition_check = Month_dict[month]['Back'].index(2) < Month_dict[month]['Back'].index(Month_dict[month]['Back'][offset])

    if front_year_condition_check == False: 
        year_actual_front = 2000 + year

    elif front_year_condition_check == True: 
        year_actual_front = 2000 + year - 1

    if back_year_condition_check==False:
        year_actual_back = 2000+year

    elif back_year_condition_check==True:
        year_actual_back = 2000 + year - 1

    back_month  = Month_dict[month]['Back'][offset]+1

    front_month = Month_dict[month]['Front'][offset]+1

    if back_month==13:
        back_month = 1
    if front_month==13:
        front_month = 1


    if back_month <= 9:
        back_separator = '-0'
    elif back_month >= 10:
        back_separator = '-'

    if front_month <= 9:
        front_separator = '-0'
    elif front_month >= 10:
        front_separator = '-'

    start_day = 1
    end_day = 1

    if start_day <=9:
        start_day_separator = '-0'
    elif start_day>=10:
        start_day_separator = '-'

    if end_day <=9:
        end_day_separator = '-0'
    elif end_day>=10:
        end_day_separator = '-'

    start_date = str(year_actual_back) + back_separator +str(back_month) + start_day_separator + str(start_day)
    end_date = str(year_actual_front) + front_separator +str(front_month) + end_day_separator + str(end_day)

    start_date = pd.to_datetime(start_date, infer_datetime_format=True)
    end_date = pd.to_datetime(end_date, infer_datetime_format=True)

    trade_mask = (tradeLog['Entry Time'] < end_date) & (tradeLog['Entry Time'] >= start_date)
    data_mask = (train.index < end_date) & (train.index >= start_date)

    revised_tradeLog = tradeLog.loc[trade_mask]
    train = train.loc[data_mask]

    return train, revised_tradeLog
