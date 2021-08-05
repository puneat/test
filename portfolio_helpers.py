import pandas as pd
import numpy as np
import warnings
from numpy import cumsum, log, polyfit, sqrt, std, subtract
%matplotlib inline
from datetime import datetime, timedelta
import scipy.stats as st
import statsmodels.api as sm
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
from functools import reduce


def prepare_portfolio(strategy_names_list, tradelog_path_list, lots_distribution_list, combine_type='day', portfolio_type = 'alone'):
    
    combined = pd.DataFrame()
    
    for i, tradelog_path in enumerate(tradelog_path_list):
        strategy = pd.read_csv(tradelog_path)
        strategy['Entry Time'] = pd.to_datetime(strategy['Entry Time'])
        strategy['Exit Time'] = pd.to_datetime(strategy['Exit Time'])
        strategy['Holding Time'] = strategy['Exit Time'] - strategy['Entry Time']
        strategy['PNL'] = strategy['PNL']*lots_distribution_list[i]
        strategy['Lots'] = strategy['Lots']*lots_distribution_list[i]
        
        if portfolio_type == 'mix':
            
            combined_list = [combined, strategy]
        
            combined = pd.concat(combined_list,axis=0)
        
            if combine_type=='year-month':
                combined = combined.groupby([combined['Exit Time'].dt.year, combined['Exit Time'].dt.month]).sum()
        
            elif combine_type=='month':
                combined = combined.groupby([combined['Exit Time'].dt.month]).sum()
            
            elif combine_type=='year':
                combined = combined.groupby([combined['Exit Time'].dt.year]).sum()
            
            elif combine_type=='day':
                combined = combined.groupby([combined['Exit Time'].dt.date]).sum()
            
            elif combine_type=='trade':
                combined  = combined
        
        elif portfolio_type == 'alone':
            
            combined_list = [strategy['PNL'], combined]

            combined = reduce(lambda  left,right: pd.merge(left,right,  left_index=True, right_index=True,
                                            how='outer'),combined_list).fillna(0)
    if portfolio_type == 'alone':
        combined.columns = strategy_names_list
            
    return combined
