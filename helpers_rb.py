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

def clean_contract_data(train, year, month, month_in_number, offset):

    front_month_list = [1,2,3,4,5,6,7,8,9,10,11,12,
                    1,2,3,4,5,6,7,8,9,10,11,12,
                    1,2,3,4,5,6,7,8,9,10,11,12]

    back_month_list = [11,12,1,2,3,4,5,6,7,8,9,10,
                   11,12,1,2,3,4,5,6,7,8,9,10,
                   11,12,1,2,3,4,5,6,7,8,9,10]

    month_in_number = month_in_number + 11

    if (month_in_number - offset) >= 12: 
        year_actual_front = 2000+year

    elif (month_in_number - offset) < 12: 
        year_actual_front = 2000 + year - 1

    if (month_in_number - offset) > 13: 
        year_actual_back = 2000+year

    elif (month_in_number - offset) <= 13:
        year_actual_back = 2000 + year - 1

    back_month  = back_month_list[month_in_number-offset]

    front_month = front_month_list[month_in_number-offset]

    if back_month <= 9:
        back_separator = '-0'
    elif back_month >= 10:
        back_separator = '-'

    if front_month <= 9:
        front_separator = '-0'
    elif front_month >= 10:
        front_separator = '-'

    if month=='Jan':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[12-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator + str(front_month_list[12-offset]) + '-0' + str(1)

    elif month=='Feb':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[13-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[13-offset]) + '-0' + str(1)

    elif month=='Mar':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[14-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[14-offset]) + '-0' + str(1)

    elif month=='Apr':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[15-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[15-offset]) + '-0' + str(1)

    elif month=='May':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[16-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[16-offset]) + '-0' + str(1)

    elif month=='Jun':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[17-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[17-offset]) + '-0' + str(1)

    elif month=='Jul':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[18-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[18-offset]) + '-0' + str(1)

    elif month=='Aug':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[19-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[19-offset]) + '-0' + str(1)

    elif month=='Sep':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[20-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[20-offset]) + '-0' + str(1)

    elif month=='Oct':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[21-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[21-offset]) + '-0' + str(1)

    elif month=='Nov':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[22-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[22-offset]) + '-0' + str(1)

    elif month=='Dec':
        start_date = str(year_actual_front) + back_separator +str(back_month_list[23-offset]) + '-' + str(10)
        end_date = str(year_actual_back) + front_separator +str(front_month_list[23-offset]) + '-0' + str(1)

    # print(start_date, end_date)

    start_date = pd.to_datetime(start_date, infer_datetime_format=True)
    end_date = pd.to_datetime(end_date, infer_datetime_format=True)
    data_mask = (train.index < end_date) & (train.index >= start_date)

    train = train.loc[data_mask]

    return train

def clean_backtest_data(train, tradeLog, year, month, month_in_number, offset):
    front_month_list = [1,2,3,4,5,6,7,8,9,10,11,12,
                    1,2,3,4,5,6,7,8,9,10,11,12,
                    1,2,3,4,5,6,7,8,9,10,11,12]

    back_month_list = [12,1,2,3,4,5,6,7,8,9,10,11,
                       12,1,2,3,4,5,6,7,8,9,10,11,
                       12,1,2,3,4,5,6,7,8,9,10,11,]

    month_in_number = month_in_number + 11

    if (month_in_number - offset) > 11: 
        year_actual_front = 2000+year

    elif (month_in_number - offset) <= 11: 
        year_actual_front = 2000 + year - 1

    if (month_in_number - offset) >= 13: 
        year_actual_back = 2000+year

    elif (month_in_number - offset) <= 12:
        year_actual_back = 2000 + year - 1

    back_month  = back_month_list[month_in_number-offset]

    front_month = front_month_list[month_in_number-offset]

    if back_month <= 9:
        back_separator = '-0'
    elif back_month >= 10:
        back_separator = '-'

    if front_month <= 9:
        front_separator = '-0'
    elif front_month >= 10:
        front_separator = '-'

    if month=='Jan':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[12-offset]) + '-0' + str(1)
        end_date = str(year_actual_front) + front_separator + str(front_month_list[12-offset]) + '-0' + str(1)

    elif month=='Feb':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[13-offset]) + '-0' + str(1)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[13-offset]) + '-0' + str(1)

    elif month=='Mar':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[14-offset]) + '-0' + str(1)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[14-offset]) + '-0' + str(1)

    elif month=='Apr':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[15-offset]) + '-0' + str(1)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[15-offset]) + '-0' + str(1)

    elif month=='May':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[16-offset]) + '-0' + str(1)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[16-offset]) + '-0' + str(1)

    elif month=='Jun':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[17-offset]) + '-0' + str(1)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[17-offset]) + '-0' + str(1)

    elif month=='Jul':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[18-offset]) + '-0' + str(1)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[18-offset]) + '-0' + str(1)

    elif month=='Aug':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[19-offset]) + '-0' + str(1)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[19-offset]) + '-0' + str(1)

    elif month=='Sep':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[20-offset]) + '-0' + str(1)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[20-offset]) + '-0' + str(1)

    elif month=='Oct':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[21-offset]) + '-0' + str(1)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[21-offset]) + '-0' + str(1)

    elif month=='Nov':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[22-offset]) + '-0' + str(1)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[22-offset]) + '-0' + str(1)

    elif month=='Dec':
        start_date = str(year_actual_front) + back_separator +str(back_month_list[23-offset]) + '-0' + str(1)
        end_date = str(year_actual_back) + front_separator +str(front_month_list[23-offset]) + '-0' + str(1)

    # print(start_date, end_date)

    start_date = pd.to_datetime(start_date, infer_datetime_format=True)
    end_date = pd.to_datetime(end_date, infer_datetime_format=True)

    trade_mask = (tradeLog['Entry Time'] <= end_date) & (tradeLog['Entry Time'] >= start_date)
    data_mask = (train.index < end_date) & (train.index >= start_date)

    revised_tradeLog = tradeLog.loc[trade_mask]
    train = train.loc[data_mask]

    return train, revised_tradeLog
