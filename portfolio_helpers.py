import pandas as pd
import numpy as np
import warnings
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from datetime import datetime, timedelta
from tqdm import tqdm
warnings.simplefilter(action='ignore',  category=Warning)
from numpy import median, mean
from functools import reduce


def prepare_portfolio(strategy_names_list, tradelog_path_list, lots_distribution_list, combine_type='day', portfolio_type = 'alone', index_type=None):
    
    combined = pd.DataFrame()
    
    for i, tradelog_path in enumerate(tradelog_path_list):
        strategy = pd.read_csv(tradelog_path)
        strategy['Entry Time'] = pd.to_datetime(strategy['Entry Time'])
        strategy['Exit Time'] = pd.to_datetime(strategy['Exit Time'])
        strategy['Holding Time'] = strategy['Exit Time'] - strategy['Entry Time']
        strategy['PNL'] = strategy['PNL']*lots_distribution_list[i]
        strategy['Lots'] = strategy['Lots']*lots_distribution_list[i]
        if index_type is not None:
            strategy = strategy.set_index(strategy[index_type])
        
        if portfolio_type == 'mix':
            
            combined_list = [combined, strategy]
        
            combined = pd.concat(combined_list,axis=0)
        
        
        elif portfolio_type == 'alone':
            
            combined_list = [strategy['PNL'], combined]

            combined = reduce(lambda  left,right: pd.merge(left,right,  left_index=True, right_index=True,
                                            how='outer'),combined_list).fillna(0)
    if portfolio_type == 'alone':
        combined.columns = strategy_names_list
        
    elif portfolio_type == 'mix':
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
            
    return combined
