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


from functools import reduce
class Broker():
    def __init__(self,
                 instrument=None,
                 strategy_obj=None,
                 min_tick_increment = 0.0001,
                 tick_value = 4.2,
                 entry_slippage_ticks = 12,
                 exit_slippage_ticks = 12,
                 default_lot_per_trade = 1,
                 use_default_lot_size = True,
                 trading_stop_day = 28,
                 overnight = True,
                 transaction_cost = 1,
                 pass_history = 1,
                 bid_data = None,
                 ask_data = None,
                 classifier = None,
                 classifier_type = 'keras'):
        
        self.instrument = instrument
        self.bid_data = bid_data
        self.ask_data = ask_data
        self.min_tick_increment = min_tick_increment
        self.tick_value = tick_value
        self.entry_slippage_ticks = entry_slippage_ticks
        self.exit_slippage_ticks = exit_slippage_ticks
        self.strategy_obj = strategy_obj
        self.trading_stop_day =  trading_stop_day
        self.overnight = overnight
        self.transaction_cost = transaction_cost
        self.pass_history = pass_history
        self.classifier = classifier
        self.classifier_type = classifier_type
        
        self.entry_price = None
        self.exit_price = None
        self.stop_price = None
        self.target_price = None
        self.position = 0
        self.pnl = 0
        self.lot_size = 0
        self.default_lot_size = default_lot_per_trade
        self.use_default_lot_size = use_default_lot_size
        
            
        self.entry_bid_open = None
        self.entry_bid_high = None
        self.entry_bid_low = None
        self.entry_bid_close = None
        self.entry_bid_volume = None
        self.exit_bid_open = None
        self.exit_bid_high = None
        self.exit_bid_low = None
        self.exit_bid_close = None
        self.exit_bid_volume = None
            
        self.entry_ask_open = None
        self.entry_ask_high = None
        self.entry_ask_low = None
        self.entry_ask_close = None
        self.entry_ask_volume = None
        self.exit_ask_open = None
        self.exit_ask_high = None
        self.exit_ask_low = None
        self.exit_ask_close = None
        self.exit_ask_volume = None
        
        self.cumulative_pnl_array = []
        self.pnl_array = []
        self.cumulative_pnl = 0
        self.trade_id = -1
        self.TSL_logs = None
        self.TSL_time_logs = None
        self.trade_type = None
        self.entry_time = None
        self.exit_time = None
        self.exit_type = None
        
        self.max_adverse_excursion = None
        
        self.max_favor_excursion = None

        self.tradeLog = pd.DataFrame(columns=['Trade ID',
                                              'Trade Type',
                                              'Entry Bid Params',
                                              'Entry Ask Params',
                                              'Entry Time',
                                              'Entry Price',
                                              'Lots',
                                              'Target Price',
                                              'TSL',
                                              'TSL time',
                                              'MFE',
                                              'MAE',
                                              'Stop Price',
                                              'Exit Bid Params',
                                              'Exit Ask Params',
                                              'Exit Time',
                                              'Exit Price',
                                              'PNL',
                                              'Holding Time',
                                              'Exit Type',
                                              'Transaction Cost',
                                               ])
            
    def tradeExit(self):
        
        self.tradeLog.loc[self.trade_id, 'Trade ID'] = self.trade_id
        
        self.tradeLog.loc[self.trade_id, 'Trade Type'] = self.trade_type
          
        self.tradeLog.loc[self.trade_id, 'Entry Bid Params'] = (round(self.entry_bid_open,4), round(self.entry_bid_high,4), round(self.entry_bid_low,4), round(self.entry_bid_close,4), self.entry_bid_volume)
            
        self.tradeLog.loc[self.trade_id, 'Entry Ask Params'] = (round(self.entry_ask_open,4), round(self.entry_ask_high,4), round(self.entry_ask_low,4), round(self.entry_ask_close,4), self.entry_ask_volume)
        
        self.tradeLog.loc[self.trade_id, 'Entry Time'] = pd.to_datetime(self.entry_time, infer_datetime_format= True)
        
        self.tradeLog.loc[self.trade_id, 'Entry Price'] = self.entry_price
        
        self.tradeLog.loc[self.trade_id, 'Lots'] = self.lot_size
        
        self.tradeLog.loc[self.trade_id, 'Target Price'] = self.target_price
 
        self.tradeLog.loc[self.trade_id, 'TSL'] = self.TSL_logs
 
        self.tradeLog.loc[self.trade_id, 'TSL time'] = self.TSL_time_logs
        
        self.tradeLog.loc[self.trade_id, 'Stop Price'] = self.stop_price
        
        self.tradeLog.loc[self.trade_id, 'Exit Bid Params'] = (round(self.exit_bid_open,4), round(self.exit_bid_high,4), round(self.exit_bid_low,4), round(self.exit_bid_close,4), self.exit_bid_volume)
            
        self.tradeLog.loc[self.trade_id, 'Exit Ask Params'] = (round(self.exit_ask_open,4), round(self.exit_ask_high,4), round(self.exit_ask_low,4), round(self.exit_ask_close,4), self.exit_ask_volume)
            
        self.tradeLog.loc[self.trade_id, 'Exit Time'] = pd.to_datetime(self.exit_time, infer_datetime_format= True)
        
        self.tradeLog.loc[self.trade_id, 'Exit Price'] = self.exit_price
        
        self.tradeLog.loc[self.trade_id, 'PNL'] = self.pnl - (self.transaction_cost * self.lot_size)
        
        self.tradeLog.loc[self.trade_id, 'Holding Time'] = (self.exit_time - self.entry_time)
            
        self.tradeLog.loc[self.trade_id, 'Exit Type'] = self.exit_type
        
        self.tradeLog.loc[self.trade_id, 'Transaction Cost'] = self.transaction_cost * self.lot_size
        
        self.tradeLog.loc[self.trade_id, 'MFE'] = abs(self.max_favor_excursion / self.min_tick_increment)
        
        self.tradeLog.loc[self.trade_id, 'MAE'] = abs(self.max_adverse_excursion / self.min_tick_increment)
        
 
    def testerAlgo(self):

        def takeEntry():
            current_month = self.bid_data.index[i].month
            current_day_of_month = self.bid_data.index[i].day

            if self.classifier_type=='keras':
                if len(self.tradeLog) > 5:

                    secondary_df = self.tradeLog
                    temp_tradelog = pd.DataFrame()

                    temp_tradelog['PNL'] = secondary_df['PNL']
                    temp_tradelog['Trade Type'] = secondary_df['Trade Type']
                    temp_tradelog['Month'] = pd.to_datetime(secondary_df['Entry Time']).dt.month
                    temp_tradelog['Entry Hour'] = pd.to_datetime(secondary_df['Entry Time']).dt.hour
                    temp_tradelog['Entry Day'] = pd.to_datetime(secondary_df['Entry Time']).dt.day
                    temp_tradelog['Exit Hour'] = pd.to_datetime(secondary_df['Exit Time']).dt.hour
                    temp_tradelog['Exit Day'] = pd.to_datetime(secondary_df['Exit Time']).dt.day
                    temp_tradelog['Target'] = np.where(secondary_df['PNL']>0,1,0)

                    data_frames = [temp_tradelog.shift(1), temp_tradelog.shift(2), temp_tradelog.shift(3), temp_tradelog.shift(4), temp_tradelog.shift(5)]

                    df_merged = reduce(lambda  left,right: pd.merge(left,right,  left_index=True, right_index=True,
                                            how='outer'), data_frames)

                    df_merged = df_merged.dropna()

                    X_live = np.asarray(df_merged.iloc[-1].values).astype(np.float32)

                    y_pred = self.classifier.predict_classes(X_live.reshape(1, -1))[0][0]

                    self.default_lot_size = y_pred + 1

            elif self.classifier_type == 'sklearn':
                if len(self.tradeLog) > 5:

                    secondary_df = self.tradeLog
                    temp_tradelog = pd.DataFrame()

                    temp_tradelog['PNL'] = secondary_df['PNL']
                    temp_tradelog['Trade Type'] = secondary_df['Trade Type']
                    temp_tradelog['Month'] = pd.to_datetime(secondary_df['Entry Time']).dt.month
                    temp_tradelog['Entry Hour'] = pd.to_datetime(secondary_df['Entry Time']).dt.hour
                    temp_tradelog['Entry Day'] = pd.to_datetime(secondary_df['Entry Time']).dt.day
                    temp_tradelog['Exit Hour'] = pd.to_datetime(secondary_df['Exit Time']).dt.hour
                    temp_tradelog['Exit Day'] = pd.to_datetime(secondary_df['Exit Time']).dt.day
                    temp_tradelog['Target'] = np.where(secondary_df['PNL']>0,1,0)

                    data_frames = [temp_tradelog.shift(1), temp_tradelog.shift(2), temp_tradelog.shift(3), temp_tradelog.shift(4), temp_tradelog.shift(5)]

                    df_merged = reduce(lambda  left,right: pd.merge(left,right,  left_index=True, right_index=True,
                                            how='outer'), data_frames)

                    df_merged = df_merged.dropna()

                    X_live = df_merged.iloc[-1].values

                    y_pred = self.classifier.predict_classes(X_live.reshape(1, -1))[0]

                    self.default_lot_size = y_pred + 1

            elif self.classifier_type == None: 
                pass


            if current_month == 2:
                current_day_of_month = current_day_of_month + 4
            if current_day_of_month <= self.trading_stop_day and ((self.bid_data.index[i].day == self.bid_data.index[i+1].day)):
                if self.pass_history =='all':
                    enterShortSignal, tmp_short_entry_price, tmp_short_target, tmp_short_stop, tmp_short_lots =  self.strategy_obj.shortEntry(self.ask_data.iloc[:i+1],self.bid_data.iloc[:i+1],
                                                                                                              self.min_tick_increment)
                
                    enterLongSignal, tmp_long_entry_price, tmp_long_target, tmp_long_stop, tmp_long_lots =  self.strategy_obj.longEntry(self.ask_data.iloc[:i+1],self.bid_data.iloc[:i+1],
                                                                                                          self.min_tick_increment)
                else:
                    assert self.pass_history%1==0
                    enterShortSignal, tmp_short_entry_price, tmp_short_target, tmp_short_stop, tmp_short_lots =  self.strategy_obj.shortEntry(self.ask_data.iloc[i-self.pass_history:i+1], self.bid_data.iloc[i-self.pass_history:i+1],
                                                                                                                                                      self.min_tick_increment)
                
                    enterLongSignal, tmp_long_entry_price, tmp_long_target, tmp_long_stop, tmp_long_lots =  self.strategy_obj.longEntry(self.ask_data.iloc[i-self.pass_history:i+1], self.bid_data.iloc[i-self.pass_history:i+1],
                                                                                                                                                self.min_tick_increment)
                if enterShortSignal == True:
                    self.position = -1
                    self.trade_id = self.trade_id + 1
                    self.trade_type = -1
                            
                    self.entry_bid_open = self.bid_data['Open'][i]
                    self.entry_bid_high = self.bid_data['High'][i]
                    self.entry_bid_low = self.bid_data['Low'][i]
                    self.entry_bid_close = self.bid_data[ 'Close'][i]
                    self.entry_bid_volume = self.bid_data['Volume'][i]
                            
                    self.entry_ask_open = self.ask_data['Open'][i]
                    self.entry_ask_high = self.ask_data['High'][i]
                    self.entry_ask_low = self.ask_data['Low'][i]
                    self.entry_ask_close = self.ask_data[ 'Close'][i]
                    self.entry_ask_volume = self.ask_data['Volume'][i]

                    self.entry_time = self.bid_data.index[i]
                    self.entry_price = round(tmp_short_entry_price - (self.min_tick_increment*self.entry_slippage_ticks),4)
                    
                    if self.use_default_lot_size:
                        self.lot_size = self.default_lot_size
                    else:
                        self.lot_size = tmp_short_lots
                        
                    self.target_price = tmp_short_target
                    self.stop_price = tmp_short_stop
                    
                elif enterLongSignal == True:
                    self.position = 1 
                    self.trade_id = self.trade_id + 1
                    self.trade_type = 1
                            
                    self.entry_bid_open = self.bid_data['Open'][i]
                    self.entry_bid_high = self.bid_data['High'][i]
                    self.entry_bid_low = self.bid_data['Low'][i]
                    self.entry_bid_close = self.bid_data[ 'Close'][i]
                    self.entry_bid_volume = self.bid_data['Volume'][i]
                            
                    self.entry_ask_open = self.ask_data['Open'][i]
                    self.entry_ask_high = self.ask_data['High'][i]
                    self.entry_ask_low = self.ask_data['Low'][i]
                    self.entry_ask_close = self.ask_data[ 'Close'][i]
                    self.entry_ask_volume = self.ask_data['Volume'][i]
                            
                    self.entry_time = self.ask_data.index[i]
                    self.entry_price = round(tmp_long_entry_price + (self.min_tick_increment*self.entry_slippage_ticks),4)
                            
                    if self.use_default_lot_size:
                        self.lot_size = self.default_lot_size
                    else:
                        self.lot_size = tmp_long_lots
                        
                    self.target_price = tmp_long_target
                    self.stop_price = tmp_long_stop 

        
        for i in (range(self.pass_history, len(self.bid_data)-1)):
            
            if self.position in [1, -1]:
                
                if self.position == -1:
                    
                    if self.max_adverse_excursion is None:
                        self.max_adverse_excursion = abs(self.bid_data['High'][i] - self.entry_price)
                    elif self.max_adverse_excursion is not None:
                        self.max_adverse_excursion = max(abs(self.bid_data['High'][i] - self.entry_price), self.max_adverse_excursion)
                        
                    if self.max_favor_excursion is None:
                        self.max_favor_excursion = abs(self.entry_price - self.bid_data['Low'][i])
                    elif self.max_adverse_excursion is not None:
                        self.max_favor_excursion = max(abs(self.entry_price - self.bid_data['Low'][i]), self.max_favor_excursion)
                    
                    if self.pass_history =='all':
                        exitShortSignal, tmp_short_exit_price,tmp_short_exit_type, tmp_short_TSL, tmp_short_TSL_time, self.stop_price, self.target_price =  self.strategy_obj.shortExit(self.ask_data.iloc[:i+1], self.bid_data.iloc[:i+1],
                                                                                                                self.stop_price, 
                                                                                                                self.target_price,
                                                                                                                self.entry_price,
                                                                                                                self.lot_size)
                    else:
                        assert self.pass_history%1==0
                        exitShortSignal, tmp_short_exit_price, tmp_short_exit_type, tmp_short_TSL, tmp_short_TSL_time,self.stop_price, self.target_price =  self.strategy_obj.shortExit(self.ask_data.iloc[i-self.pass_history:i+1], self.bid_data.iloc[i-self.pass_history:i+1],
                                                                                                                self.stop_price, 
                                                                                                                self.target_price,
                                                                                                                self.entry_price,
                                                                                                                self.lot_size)
                    if exitShortSignal == True:
                        self.position = 0
                        
                        
                        self.exit_price = round(tmp_short_exit_price + (self.min_tick_increment*self.exit_slippage_ticks),4)
                        if tmp_short_exit_type in ['Target', 'Stop', 'Extra']:
                            self.pnl = ((self.entry_price - self.exit_price)/self.min_tick_increment)*self.tick_value*self.lot_size
                        else:
                            assert tmp_short_exit_type in ['Target', 'Stop', 'Extra']
                            
                        self.exit_type = tmp_short_exit_type
                        
                        self.cumulative_pnl = self.cumulative_pnl + self.pnl
                        self.cumulative_pnl_array.append(self.cumulative_pnl)
                        self.pnl_array.append(self.pnl)
                        self.exit_time = self.ask_data.index[i]
                            
                        self.exit_bid_open = self.bid_data['Open'][i]
                        self.exit_bid_high = self.bid_data['High'][i]
                        self.exit_bid_low = self.bid_data['Low'][i]
                        self.exit_bid_close = self.bid_data[ 'Close'][i]
                        self.exit_bid_volume = self.bid_data['Volume'][i]
                            
                        self.exit_ask_open = self.ask_data['Open'][i]
                        self.exit_ask_high = self.ask_data['High'][i]
                        self.exit_ask_low = self.ask_data['Low'][i]
                        self.exit_ask_close = self.ask_data[ 'Close'][i]
                        self.exit_ask_volume = self.ask_data['Volume'][i]
                        self.TSL_logs = tmp_short_TSL
                        self.TSL_time_logs = tmp_short_TSL_time
                        
                        self.tradeExit()
                        self.max_adverse_excursion = None
                        self.max_favor_excursion = None
                        takeEntry()
                        
                if self.position == 1:
                    
                    if self.max_adverse_excursion is None:
                        self.max_adverse_excursion = abs(self.entry_price - self.bid_data['Low'][i])
                    elif self.max_adverse_excursion is not None:
                        self.max_adverse_excursion = max(abs(self.entry_price - self.bid_data['Low'][i]),self.max_adverse_excursion)
                        
                    if self.max_favor_excursion is None:
                        self.max_favor_excursion = abs(self.bid_data['High'][i] - self.entry_price)
                    elif self.max_adverse_excursion is not None:
                        self.max_favor_excursion = max(abs(self.bid_data['High'][i] - self.entry_price), self.max_favor_excursion)
                    
                    if self.pass_history =='all':
                        exitLongSignal, tmp_long_exit_price, tmp_long_exit_type, tmp_long_TSL, tmp_long_TSL_time, self.stop_price, self.target_price =  self.strategy_obj.longExit(self.ask_data.iloc[:i+1], self.bid_data.iloc[:i+1],
                                                                                                              self.stop_price, 
                                                                                                              self.target_price,
                                                                                                              self.entry_price,
                                                                                                              self.lot_size)
                    else:
                        assert self.pass_history%1==0
                        exitLongSignal, tmp_long_exit_price, tmp_long_exit_type, tmp_long_TSL, tmp_long_TSL_time, self.stop_price, self.target_price =  self.strategy_obj.longExit(self.ask_data.iloc[i-self.pass_history:i+1],self.bid_data.iloc[i-self.pass_history:i+1],
                                                                                                              self.stop_price, 
                                                                                                              self.target_price,
                                                                                                              self.entry_price,
                                                                                                              self.lot_size)
                        
                    if exitLongSignal == True:
                        self.position = 0
                        
                        self.exit_price = round(tmp_long_exit_price - (self.min_tick_increment*self.exit_slippage_ticks),4)
                        if tmp_long_exit_type in ['Target', 'Stop', 'Extra']:
                            self.pnl = ((self.exit_price - self.entry_price)/self.min_tick_increment)*self.tick_value*self.lot_size
                        else:
                            assert tmp_long_exit_type in ['Target', 'Stop', 'Extra']
                            
                        self.exit_type = tmp_long_exit_type
                        
                        self.cumulative_pnl = self.cumulative_pnl + self.pnl
                        self.cumulative_pnl_array.append(self.cumulative_pnl)
                        self.pnl_array.append(self.pnl)
                        self.exit_time = self.bid_data.index[i]
                            
                        self.exit_bid_open = self.bid_data['Open'][i]
                        self.exit_bid_high = self.bid_data['High'][i]
                        self.exit_bid_low = self.bid_data['Low'][i]
                        self.exit_bid_close = self.bid_data[ 'Close'][i]
                        self.exit_bid_volume = self.bid_data['Volume'][i]
                            
                        self.exit_ask_open = self.ask_data['Open'][i]
                        self.exit_ask_high = self.ask_data['High'][i]
                        self.exit_ask_low = self.ask_data['Low'][i]
                        self.exit_ask_close = self.ask_data[ 'Close'][i]
                        self.exit_ask_volume = self.ask_data['Volume'][i]
 
                        self.TSL_logs = tmp_long_TSL
                        self.TSL_time_logs = tmp_long_TSL_time
                        
                        self.tradeExit()
                        self.max_adverse_excursion = None
                        self.max_favor_excursion = None
                        takeEntry()
                        
                
                current_month = self.bid_data.index[i].month
                current_day_of_month = self.bid_data.index[i].day
                
                if current_month == 2:
                    current_day_of_month = current_day_of_month + 4
                    
                if current_day_of_month >= self.trading_stop_day+1:
                    
                    if self.position == 1:
                        self.exit_price = self.bid_data['Close'][i]
                            
                    elif self.position == -1:
                        self.exit_price = self.ask_data['Close'][i]
                    
                    if self.position == 1:
                        self.pnl = ((self.exit_price - self.entry_price - (self.min_tick_increment*self.exit_slippage_ticks))/self.min_tick_increment)*self.tick_value*self.lot_size
                    elif self.position == -1:
                        self.pnl = ((self.entry_price - self.exit_price + (self.min_tick_increment*self.exit_slippage_ticks))/self.min_tick_increment)*self.tick_value*self.lot_size
                            
                    self.exit_type = 'Expiry'
                        
                    self.cumulative_pnl = self.cumulative_pnl + self.pnl
                    self.cumulative_pnl_array.append(self.cumulative_pnl)
                    self.pnl_array.append(self.pnl)
                    self.exit_time = self.bid_data.index[i]
                        
                    self.exit_bid_open = self.bid_data['Open'][i]
                    self.exit_bid_high = self.bid_data['High'][i]
                    self.exit_bid_low = self.bid_data['Low'][i]
                    self.exit_bid_close = self.bid_data[ 'Close'][i]
                    self.exit_bid_volume = self.bid_data['Volume'][i]
                            
                    self.exit_ask_open = self.ask_data['Open'][i]
                    self.exit_ask_high = self.ask_data['High'][i]
                    self.exit_ask_low = self.ask_data['Low'][i]
                    self.exit_ask_close = self.ask_data[ 'Close'][i]
                    self.exit_ask_volume = self.ask_data['Volume'][i]
                        
                    self.position = 0
                        
                    self.tradeExit()
                    
                if self.overnight == False:
                    
                    if (self.bid_data.index[i].day != self.bid_data.index[i+1].day) or (self.bid_data.index[i].month != self.bid_data.index[i+1].month):
                    
                        if self.position == 1:
                            self.exit_price = self.bid_data['Close'][i]
                            
                        elif self.position == -1:
                            self.exit_price = self.ask_data['Close'][i]
                    
                        if self.position == 1:
                            self.pnl = ((self.exit_price - self.entry_price - (self.min_tick_increment*self.exit_slippage_ticks))/self.min_tick_increment)*self.tick_value*self.lot_size

                        elif self.position == -1:
                            self.pnl = ((self.entry_price - self.exit_price + (self.min_tick_increment*self.exit_slippage_ticks))/self.min_tick_increment)*self.tick_value*self.lot_size
                            
                        self.exit_type = 'Overnight Close'
                        
                        self.cumulative_pnl = self.cumulative_pnl + self.pnl
                        self.cumulative_pnl_array.append(self.cumulative_pnl)
                        self.pnl_array.append(self.pnl)
                        self.exit_time = self.bid_data.index[i]
                            
                        self.exit_bid_open = self.bid_data['Open'][i]
                        self.exit_bid_high = self.bid_data['High'][i]
                        self.exit_bid_low = self.bid_data['Low'][i]
                        self.exit_bid_close = self.bid_data[ 'Close'][i]
                        self.exit_bid_volume = self.bid_data['Volume'][i]
                            
                        self.exit_ask_open = self.ask_data['Open'][i]
                        self.exit_ask_high = self.ask_data['High'][i]
                        self.exit_ask_low = self.ask_data['Low'][i]
                        self.exit_ask_close = self.ask_data[ 'Close'][i]
                        self.exit_ask_volume = self.ask_data['Volume'][i]
                        self.position = 0
                        
                        self.tradeExit()

            elif self.position == 0:
                takeEntry()

class Metrics():
    def __init__(self, 
                 trade_logs,
                 min_tick_increment = 0.0001,
                 tick_value = 4.2,
                 slippage_ticks = 0,
                 transaction_costs = 1,
                 risk_free_rate = 1,
                 metric_category = 'Strategy'):
        
        self.trade_logs = trade_logs
        self.min_tick_increment = min_tick_increment
        self.tick_value = tick_value
        self.slippage_ticks = slippage_ticks
        self.transaction_costs = transaction_costs
        self.risk_free_rate = risk_free_rate
        self.trade_logs['Entry Time'] = pd.to_datetime(self.trade_logs['Entry Time'], infer_datetime_format= True)
        self.trade_logs['Exit Time'] = pd.to_datetime(self.trade_logs['Exit Time'], infer_datetime_format= True)
        self.metric_category = metric_category
        
        self.performance_metrics = pd.DataFrame(index=['Total Trades',
        'Winning Trades',
        'Losing Trades',
        'Net P/L',
        'Gross Profit',
        'Gross Loss',
        'Max Profit',
        'Max Loss',
        'Min Profit', 
        'Min Loss',
        'Max Holding Time',
        'Min Holding Time',
        'Avg Holding Time',
        'P/L Per Trade',
        'Max Drawdown',
        'Max Drawdown Duration',
        'Win Percentage',
        'Profit Factor',
        'Magic Number',
        'Profit Per Winning Trade',
        'Loss Per Losing Trade',
        'P/L Per Lot',
        'Gross Transaction Costs',
        'Long Net P/L', 
        'Short Net P/L'])
        
        self.monthly_performance = pd.DataFrame()
        
        self.yearly_performance = pd.DataFrame()
        
        self.weekly_performance = pd.DataFrame()
        
        self.hourly_entry_performance = pd.DataFrame()
        
        self.hourly_exit_performance = pd.DataFrame()

        self.long_trade_logs = self.trade_logs.loc[self.trade_logs['Trade Type']==1].reset_index()

        self.short_trade_logs = self.trade_logs.loc[self.trade_logs['Trade Type']==-1].reset_index()
        
    def overall_calc(self):
        
            def total_trades_calc(self):
                return len(self.trade_logs)
    
            self.performance_metrics.loc['Total Trades', (self.metric_category+'--Overall')] = total_trades_calc(self)
            ################################################
            def winning_trades_calc(self):
                mask  = self.trade_logs['PNL']>0
                return len(self.trade_logs.loc[mask])
        
            self.performance_metrics.loc['Winning Trades', (self.metric_category+'--Overall')] = winning_trades_calc(self)
            ################################################
            def losing_trades_calc(self):
                mask  = self.trade_logs['PNL']<0
                return len(self.trade_logs.loc[mask])
        
            self.performance_metrics.loc['Losing Trades', (self.metric_category+'--Overall')] = losing_trades_calc(self)
            ################################################
            def gross_profit_calc(self):
                mask  = self.trade_logs['PNL']>0
                if len(self.trade_logs.loc[mask])>0:
                    return round(sum(self.trade_logs['PNL'].loc[mask]),2)
                else:
                    return 0
        
            self.performance_metrics.loc['Gross Profit', (self.metric_category+'--Overall')] = gross_profit_calc(self)
            ################################################
            def gross_loss_calc(self):
                mask  = self.trade_logs['PNL']<0
                if len(self.trade_logs.loc[mask])>0:
                    return round(sum(self.trade_logs['PNL'].loc[mask]),2)
                else:
                    return 0
        
            self.performance_metrics.loc['Gross Loss', (self.metric_category+'--Overall')] = gross_loss_calc(self)
            ################################################
            def net_pnl_calc(self):
                return round(sum(self.trade_logs['PNL']),2)
        
            self.performance_metrics.loc['Net P/L', (self.metric_category+'--Overall')] = net_pnl_calc(self)
            ################################################
            def max_profit_calc(self):
                mask  = self.trade_logs['PNL']>0
                return round(max(self.trade_logs['PNL']),2)
        
            self.performance_metrics.loc['Max Profit', (self.metric_category+'--Overall')] = max_profit_calc(self)
            ################################################
            def max_loss_calc(self):
                mask  = self.trade_logs['PNL']<0
                return round(min(self.trade_logs['PNL']),2)
        
            self.performance_metrics.loc['Max Loss', (self.metric_category+'--Overall')] = max_loss_calc(self)
            ################################################
            def min_profit_calc(self):
                mask  = self.trade_logs['PNL']>0
                if len(self.trade_logs.loc[mask])>0:
                    return round(min(self.trade_logs['PNL'].loc[mask]),2)
                else:
                    return np.nan
        
            self.performance_metrics.loc['Min Profit', (self.metric_category+'--Overall')] = min_profit_calc(self)
            ################################################
            def min_loss_calc(self):
                mask  = self.trade_logs['PNL']<0
                if len(self.trade_logs.loc[mask])>0:
                    return round(max(self.trade_logs['PNL'].loc[mask]),2)
                else:
                    return np.nan
        
            self.performance_metrics.loc['Min Loss', (self.metric_category+'--Overall')] = min_loss_calc(self)
            ################################################
            def pnl_per_trade_calc(self):
                return round(sum(self.trade_logs['PNL'])/len(self.trade_logs), 3)
        
            self.performance_metrics.loc['P/L Per Trade', (self.metric_category+'--Overall')] = pnl_per_trade_calc(self)
            ################################################
            def max_holding_time_calc(self):
                return max(self.trade_logs['Holding Time'])
        
            self.performance_metrics.loc['Max Holding Time', (self.metric_category+'--Overall')] = max_holding_time_calc(self)
            ################################################
            def min_holding_time_calc(self):
                return min(self.trade_logs['Holding Time'])
        
            self.performance_metrics.loc['Min Holding Time', (self.metric_category+'--Overall')] = min_holding_time_calc(self)
            ################################################
            def avg_holding_time_calc(self):
                return sum(self.trade_logs['Holding Time'], timedelta())/len(self.trade_logs)
        
            self.performance_metrics.loc['Avg Holding Time', (self.metric_category+'--Overall')] = avg_holding_time_calc(self)
            ################################################
            def win_percentage_calc(self):
                return round((self.performance_metrics.loc['Winning Trades', (self.metric_category+'--Overall')]/self.performance_metrics.loc['Total Trades', (self.metric_category+'--Overall')])*100,2)
        
            self.performance_metrics.loc['Win Percentage', (self.metric_category+'--Overall')] = win_percentage_calc(self)
            ################################################
            def profit_factor_calc(self):
                return round(abs(self.performance_metrics.loc['Gross Profit', (self.metric_category+'--Overall')]/self.performance_metrics.loc['Gross Loss', (self.metric_category+'--Overall')]), 2)
        
            self.performance_metrics.loc['Profit Factor', (self.metric_category+'--Overall')] = profit_factor_calc(self)
            ################################################

            def pnl_per_lot_calc(self):
            
                return round(sum(self.trade_logs['PNL'])/sum(self.trade_logs['Lots']), 3)
        
            self.performance_metrics.loc['P/L Per Lot', (self.metric_category+'--Overall')] = pnl_per_lot_calc(self)
            ################################################
            def pnl_per_win_calc(self):
                # mask  = self.trade_logs['PNL']>0
                # if len(self.trade_logs.loc[mask])>0:
                #     return round(mean(self.trade_logs['PNL'].loc[mask]),2)
                # else:
                #     return np.nan
                return round((self.performance_metrics.loc['Gross Profit', (self.metric_category+'--Overall')]/self.performance_metrics.loc['Winning Trades', (self.metric_category+'--Overall')]),2)
        
            self.performance_metrics.loc['Profit Per Winning Trade', (self.metric_category+'--Overall')] = pnl_per_win_calc(self)
            ################################################
            def pnl_per_loss_calc(self):
                return round((self.performance_metrics.loc['Gross Loss', (self.metric_category+'--Overall')]/self.performance_metrics.loc['Losing Trades', (self.metric_category+'--Overall')]),2)
        
            self.performance_metrics.loc['Loss Per Losing Trade', (self.metric_category+'--Overall')] = pnl_per_loss_calc(self)
            ################################################
            def transaction_cost(self):
                return sum(self.trade_logs['Transaction Cost'])
        
            self.performance_metrics.loc['Gross Transaction Costs', (self.metric_category+'--Overall')] = transaction_cost(self)
            ################################################
            def max_drawdown_calc(self):
                xs = self.trade_logs['PNL'].cumsum() # start of drawdown
                i = np.argmax(np.maximum.accumulate(xs) - xs) # start of drawdown
                j = np.argmax(xs[:i])# end of drawdown
                return round(abs(xs[i]-xs[j]),2)
        
            self.performance_metrics.loc['Max Drawdown', (self.metric_category+'--Overall')] = max_drawdown_calc(self)
            ################################################
            def max_drawdown_duration_calc(self):
                xs = self.trade_logs['PNL'].cumsum() # start of drawdown
                i = np.argmax(np.maximum.accumulate(xs) - xs) # start of drawdown
                j = np.argmax(xs[:i])# end of drawdown

                return (self.trade_logs.loc[i,'Entry Time'] - self.trade_logs.loc[j,'Entry Time']).days
        
            self.performance_metrics.loc['Max Drawdown Duration', (self.metric_category+'--Overall')] = max_drawdown_duration_calc(self)
            ###############################################

            def magic_number_calc(self):
                return round(((self.performance_metrics.loc['Profit Per Winning Trade', (self.metric_category+'--Overall')]*self.performance_metrics.loc['Win Percentage', (self.metric_category+'--Overall')]/100) + 
                             (self.performance_metrics.loc['Loss Per Losing Trade', (self.metric_category+'--Overall')]*(1-(self.performance_metrics.loc['Win Percentage', (self.metric_category+'--Overall')]/100)))), 2)
        
            self.performance_metrics.loc['Magic Number', (self.metric_category+'--Overall')] = magic_number_calc(self)
            ################################################

            def monthly_perf_calc(self):
                return self.trade_logs.groupby(self.trade_logs['Entry Time'].dt.month)['PNL'].sum()
            
            self.monthly_performance[(self.metric_category+'--Overall')] = monthly_perf_calc(self)
            ###############################################
            def yearly_perf_calc(self):
                return self.trade_logs.groupby(self.trade_logs['Entry Time'].dt.year)['PNL'].sum()
            
            self.yearly_performance[(self.metric_category+'--Overall')] = yearly_perf_calc(self)
            ################################################
            def weekly_perf_calc(self):
                return self.trade_logs.groupby(self.trade_logs['Exit Time'].dt.dayofweek)['PNL'].sum()
            
            self.weekly_performance[(self.metric_category+'--Overall')] = weekly_perf_calc(self)
            ################################################
            def hourly_entry_perf_calc(self):
                return self.trade_logs.loc[self.trade_logs['PNL']>0].groupby(self.trade_logs['Entry Time'].dt.hour)['PNL'].count()
            
            self.hourly_entry_performance[(self.metric_category+'--Overall')] = hourly_entry_perf_calc(self)
        
            ################################################
            def hourly_exit_perf_calc(self):
                return self.trade_logs.groupby(self.trade_logs['Exit Time'].dt.hour)['PNL'].apply(np.mean)
            
            self.hourly_exit_performance[(self.metric_category+'--Overall')] = hourly_exit_perf_calc(self)

            ################################################
        
    def long_calc(self):
        
            def total_trades_calc(self):
                return len(self.long_trade_logs)
    
            self.performance_metrics.loc['Total Trades', (self.metric_category+'--Long')] = total_trades_calc(self)
            ################################################
            def winning_trades_calc(self):
                mask  = self.long_trade_logs['PNL']>0
                return len(self.long_trade_logs.loc[mask])
        
            self.performance_metrics.loc['Winning Trades', (self.metric_category+'--Long')] = winning_trades_calc(self)
            ################################################
            def losing_trades_calc(self):
                mask  = self.long_trade_logs['PNL']<0
                return len(self.long_trade_logs.loc[mask])
        
            self.performance_metrics.loc['Losing Trades', (self.metric_category+'--Long')] = losing_trades_calc(self)
            ################################################
            def gross_profit_calc(self):
                mask  = self.long_trade_logs['PNL']>0
                if len(self.long_trade_logs.loc[mask])>0:
                    return round(sum(self.long_trade_logs['PNL'].loc[mask]),2)
                else:
                    return 0
        
            self.performance_metrics.loc['Gross Profit', (self.metric_category+'--Long')] = gross_profit_calc(self)
            ################################################
            def gross_loss_calc(self):
                mask  = self.long_trade_logs['PNL']<0
                if len(self.long_trade_logs.loc[mask])>0:
                    return round(sum(self.long_trade_logs['PNL'].loc[mask]),2)
                else:
                    return 0
        
            self.performance_metrics.loc['Gross Loss', (self.metric_category+'--Long')] = gross_loss_calc(self)
            ################################################
            def net_pnl_calc(self):
                return round(sum(self.long_trade_logs['PNL']),2)
        
            self.performance_metrics.loc['Net P/L', (self.metric_category+'--Long')] = net_pnl_calc(self)
            ################################################
            def max_profit_calc(self):
                mask  = self.long_trade_logs['PNL']>0
                return round(max(self.long_trade_logs['PNL']),2)
        
            self.performance_metrics.loc['Max Profit', (self.metric_category+'--Long')] = max_profit_calc(self)
            ################################################
            def max_loss_calc(self):
                mask  = self.long_trade_logs['PNL']<0
                return round(min(self.long_trade_logs['PNL']),2)
        
            self.performance_metrics.loc['Max Loss', (self.metric_category+'--Long')] = max_loss_calc(self)
            ################################################
            def min_profit_calc(self):
                mask  = self.long_trade_logs['PNL']>0
                if len(self.long_trade_logs.loc[mask])>0:
                    return round(min(self.long_trade_logs['PNL'].loc[mask]),2)
                else:
                    return np.nan
        
            self.performance_metrics.loc['Min Profit', (self.metric_category+'--Long')] = min_profit_calc(self)
            ################################################
            def min_loss_calc(self):
                mask  = self.long_trade_logs['PNL']<0
                if len(self.long_trade_logs.loc[mask])>0:
                    return round(max(self.long_trade_logs['PNL'].loc[mask]),2)
                else:
                    return np.nan
        
            self.performance_metrics.loc['Min Loss', (self.metric_category+'--Long')] = min_loss_calc(self)
            ################################################
            def pnl_per_trade_calc(self):
                return round(sum(self.long_trade_logs['PNL'])/len(self.long_trade_logs), 3)
        
            self.performance_metrics.loc['P/L Per Trade', (self.metric_category+'--Long')] = pnl_per_trade_calc(self)
            ################################################
            def max_holding_time_calc(self):
                return max(self.long_trade_logs['Holding Time'])
        
            self.performance_metrics.loc['Max Holding Time', (self.metric_category+'--Long')] = max_holding_time_calc(self)
            ################################################
            def min_holding_time_calc(self):
                return min(self.long_trade_logs['Holding Time'])
        
            self.performance_metrics.loc['Min Holding Time', (self.metric_category+'--Long')] = min_holding_time_calc(self)
            ################################################
            def avg_holding_time_calc(self):
                return sum(self.long_trade_logs['Holding Time'], timedelta())/len(self.long_trade_logs)
        
            self.performance_metrics.loc['Avg Holding Time', (self.metric_category+'--Long')] = avg_holding_time_calc(self)
            ################################################
            def win_percentage_calc(self):
                return round((self.performance_metrics.loc['Winning Trades', (self.metric_category+'--Long')]/self.performance_metrics.loc['Total Trades', (self.metric_category+'--Long')])*100,2)
        
            self.performance_metrics.loc['Win Percentage', (self.metric_category+'--Long')] = win_percentage_calc(self)
            ################################################
            def profit_factor_calc(self):
                return round(abs(self.performance_metrics.loc['Gross Profit', (self.metric_category+'--Long')]/self.performance_metrics.loc['Gross Loss', (self.metric_category+'--Long')]), 2)
        
            self.performance_metrics.loc['Profit Factor', (self.metric_category+'--Long')] = profit_factor_calc(self)
            ################################################

            def pnl_per_lot_calc(self):
            
                return round(sum(self.long_trade_logs['PNL'])/sum(self.long_trade_logs['Lots']), 3)
        
            self.performance_metrics.loc['P/L Per Lot', (self.metric_category+'--Long')] = pnl_per_lot_calc(self)
            ################################################
            def pnl_per_win_calc(self):
                # mask  = self.long_trade_logs['PNL']>0
                # if len(self.long_trade_logs.loc[mask])>0:
                #     return round(mean(self.long_trade_logs['PNL'].loc[mask]),2)
                # else:
                #     return np.nan
                return round((self.performance_metrics.loc['Gross Profit', (self.metric_category+'--Long')]/self.performance_metrics.loc['Winning Trades', (self.metric_category+'--Long')]),2)
        
            self.performance_metrics.loc['Profit Per Winning Trade', (self.metric_category+'--Long')] = pnl_per_win_calc(self)
            ################################################
            def pnl_per_loss_calc(self):
                return round((self.performance_metrics.loc['Gross Loss', (self.metric_category+'--Long')]/self.performance_metrics.loc['Losing Trades', (self.metric_category+'--Long')]),2)
        
            self.performance_metrics.loc['Loss Per Losing Trade', (self.metric_category+'--Long')] = pnl_per_loss_calc(self)
            ################################################
            def transaction_cost(self):
                return sum(self.long_trade_logs['Transaction Cost'])
        
            self.performance_metrics.loc['Gross Transaction Costs', (self.metric_category+'--Long')] = transaction_cost(self)
            ################################################
            def max_drawdown_calc(self):
                xs = self.long_trade_logs['PNL'].cumsum() # start of drawdown
                i = np.argmax(np.maximum.accumulate(xs) - xs) # start of drawdown
                j = np.argmax(xs[:i])# end of drawdown
                return round(abs(xs[i]-xs[j]),2)
        
            self.performance_metrics.loc['Max Drawdown', (self.metric_category+'--Long')] = max_drawdown_calc(self)
            ################################################
            def max_drawdown_duration_calc(self):
                xs = self.long_trade_logs['PNL'].cumsum() # start of drawdown
                i = np.argmax(np.maximum.accumulate(xs) - xs) # start of drawdown
                j = np.argmax(xs[:i])# end of drawdown

                return (self.long_trade_logs.loc[i,'Entry Time'] - self.long_trade_logs.loc[j,'Entry Time']).days
        
            self.performance_metrics.loc['Max Drawdown Duration', (self.metric_category+'--Long')] = max_drawdown_duration_calc(self)
            ###############################################

            def magic_number_calc(self):
                return round(((self.performance_metrics.loc['Profit Per Winning Trade', (self.metric_category+'--Long')]*self.performance_metrics.loc['Win Percentage', (self.metric_category+'--Long')]/100) + 
                             (self.performance_metrics.loc['Loss Per Losing Trade', (self.metric_category+'--Long')]*(1-(self.performance_metrics.loc['Win Percentage', (self.metric_category+'--Long')]/100)))), 2)
        
            self.performance_metrics.loc['Magic Number', (self.metric_category+'--Long')] = magic_number_calc(self)
            ################################################

            def monthly_perf_calc(self):
                return self.long_trade_logs.groupby(self.long_trade_logs['Entry Time'].dt.month)['PNL'].sum()
            
            self.monthly_performance[(self.metric_category+'--Long')] = monthly_perf_calc(self)
            ###############################################
            def yearly_perf_calc(self):
                return self.long_trade_logs.groupby(self.long_trade_logs['Entry Time'].dt.year)['PNL'].sum()
            
            self.yearly_performance[(self.metric_category+'--Long')] = yearly_perf_calc(self)
            ################################################
            def weekly_perf_calc(self):
                return self.long_trade_logs.groupby(self.long_trade_logs['Exit Time'].dt.dayofweek)['PNL'].sum()
            
            self.weekly_performance[(self.metric_category+'--Long')] = weekly_perf_calc(self)
            ################################################
            def hourly_entry_perf_calc(self):
                return self.long_trade_logs.loc[self.long_trade_logs['PNL']>0].groupby(self.long_trade_logs['Entry Time'].dt.hour)['PNL'].count()
            
            self.hourly_entry_performance[(self.metric_category+'--Long')] = hourly_entry_perf_calc(self)
        
            ################################################
            def hourly_exit_perf_calc(self):
                return self.long_trade_logs.groupby(self.long_trade_logs['Exit Time'].dt.hour)['PNL'].apply(np.mean)
            
            self.hourly_exit_performance[(self.metric_category+'--Long')] = hourly_exit_perf_calc(self)

            ################################################

    def short_calc(self):
        
            def total_trades_calc(self):
                return len(self.short_trade_logs)
    
            self.performance_metrics.loc['Total Trades', (self.metric_category+'--Short')] = total_trades_calc(self)
            ################################################
            def winning_trades_calc(self):
                mask  = self.short_trade_logs['PNL']>0
                return len(self.short_trade_logs.loc[mask])
        
            self.performance_metrics.loc['Winning Trades', (self.metric_category+'--Short')] = winning_trades_calc(self)
            ################################################
            def losing_trades_calc(self):
                mask  = self.short_trade_logs['PNL']<0
                return len(self.short_trade_logs.loc[mask])
        
            self.performance_metrics.loc['Losing Trades', (self.metric_category+'--Short')] = losing_trades_calc(self)
            ################################################
            def gross_profit_calc(self):
                mask  = self.short_trade_logs['PNL']>0
                if len(self.short_trade_logs.loc[mask])>0:
                    return round(sum(self.short_trade_logs['PNL'].loc[mask]),2)
                else:
                    return 0
        
            self.performance_metrics.loc['Gross Profit', (self.metric_category+'--Short')] = gross_profit_calc(self)
            ################################################
            def gross_loss_calc(self):
                mask  = self.short_trade_logs['PNL']<0
                if len(self.short_trade_logs.loc[mask])>0:
                    return round(sum(self.short_trade_logs['PNL'].loc[mask]),2)
                else:
                    return 0
        
            self.performance_metrics.loc['Gross Loss', (self.metric_category+'--Short')] = gross_loss_calc(self)
            ################################################
            def net_pnl_calc(self):
                return round(sum(self.short_trade_logs['PNL']),2)
        
            self.performance_metrics.loc['Net P/L', (self.metric_category+'--Short')] = net_pnl_calc(self)
            ################################################
            def max_profit_calc(self):
                mask  = self.short_trade_logs['PNL']>0
                return round(max(self.short_trade_logs['PNL']),2)
        
            self.performance_metrics.loc['Max Profit', (self.metric_category+'--Short')] = max_profit_calc(self)
            ################################################
            def max_loss_calc(self):
                mask  = self.short_trade_logs['PNL']<0
                return round(min(self.short_trade_logs['PNL']),2)
        
            self.performance_metrics.loc['Max Loss', (self.metric_category+'--Short')] = max_loss_calc(self)
            ################################################
            def min_profit_calc(self):
                mask  = self.short_trade_logs['PNL']>0
                if len(self.short_trade_logs.loc[mask])>0:
                    return round(min(self.short_trade_logs['PNL'].loc[mask]),2)
                else:
                    return np.nan
        
            self.performance_metrics.loc['Min Profit', (self.metric_category+'--Short')] = min_profit_calc(self)
            ################################################
            def min_loss_calc(self):
                mask  = self.short_trade_logs['PNL']<0
                if len(self.short_trade_logs.loc[mask])>0:
                    return round(max(self.short_trade_logs['PNL'].loc[mask]),2)
                else:
                    return np.nan
        
            self.performance_metrics.loc['Min Loss', (self.metric_category+'--Short')] = min_loss_calc(self)
            ################################################
            def pnl_per_trade_calc(self):
                return round(sum(self.short_trade_logs['PNL'])/len(self.short_trade_logs), 3)
        
            self.performance_metrics.loc['P/L Per Trade', (self.metric_category+'--Short')] = pnl_per_trade_calc(self)
            ################################################
            def max_holding_time_calc(self):
                return max(self.short_trade_logs['Holding Time'])
        
            self.performance_metrics.loc['Max Holding Time', (self.metric_category+'--Short')] = max_holding_time_calc(self)
            ################################################
            def min_holding_time_calc(self):
                return min(self.short_trade_logs['Holding Time'])
        
            self.performance_metrics.loc['Min Holding Time', (self.metric_category+'--Short')] = min_holding_time_calc(self)
            ################################################
            def avg_holding_time_calc(self):
                return sum(self.short_trade_logs['Holding Time'], timedelta())/len(self.short_trade_logs)
        
            self.performance_metrics.loc['Avg Holding Time', (self.metric_category+'--Short')] = avg_holding_time_calc(self)
            ################################################
            def win_percentage_calc(self):
                return round((self.performance_metrics.loc['Winning Trades', (self.metric_category+'--Short')]/self.performance_metrics.loc['Total Trades', (self.metric_category+'--Short')])*100,2)
        
            self.performance_metrics.loc['Win Percentage', (self.metric_category+'--Short')] = win_percentage_calc(self)
            ################################################
            def profit_factor_calc(self):
                return round(abs(self.performance_metrics.loc['Gross Profit', (self.metric_category+'--Short')]/self.performance_metrics.loc['Gross Loss', (self.metric_category+'--Short')]), 2)
        
            self.performance_metrics.loc['Profit Factor', (self.metric_category+'--Short')] = profit_factor_calc(self)
            ################################################

            def pnl_per_lot_calc(self):
            
                return round(sum(self.short_trade_logs['PNL'])/sum(self.short_trade_logs['Lots']), 3)
        
            self.performance_metrics.loc['P/L Per Lot', (self.metric_category+'--Short')] = pnl_per_lot_calc(self)
            ################################################
            def pnl_per_win_calc(self):
                # mask  = self.short_trade_logs['PNL']>0
                # if len(self.short_trade_logs.loc[mask])>0:
                #     return round(mean(self.short_trade_logs['PNL'].loc[mask]),2)
                # else:
                #     return np.nan
                return round((self.performance_metrics.loc['Gross Profit', (self.metric_category+'--Short')]/self.performance_metrics.loc['Winning Trades', (self.metric_category+'--Short')]),2)
        
            self.performance_metrics.loc['Profit Per Winning Trade', (self.metric_category+'--Short')] = pnl_per_win_calc(self)
            ################################################
            def pnl_per_loss_calc(self):
                return round((self.performance_metrics.loc['Gross Loss', (self.metric_category+'--Short')]/self.performance_metrics.loc['Losing Trades', (self.metric_category+'--Short')]),2)
        
            self.performance_metrics.loc['Loss Per Losing Trade', (self.metric_category+'--Short')] = pnl_per_loss_calc(self)
            ################################################
            def transaction_cost(self):
                return sum(self.short_trade_logs['Transaction Cost'])
        
            self.performance_metrics.loc['Gross Transaction Costs', (self.metric_category+'--Short')] = transaction_cost(self)
            ################################################
            def max_drawdown_calc(self):
                xs = self.short_trade_logs['PNL'].cumsum() # start of drawdown
                i = np.argmax(np.maximum.accumulate(xs) - xs) # start of drawdown
                j = np.argmax(xs[:i])# end of drawdown
                return round(abs(xs[i]-xs[j]),2)
        
            self.performance_metrics.loc['Max Drawdown', (self.metric_category+'--Short')] = max_drawdown_calc(self)
            ################################################
            def max_drawdown_duration_calc(self):
                xs = self.short_trade_logs['PNL'].cumsum() # start of drawdown
                i = np.argmax(np.maximum.accumulate(xs) - xs) # start of drawdown
                j = np.argmax(xs[:i])# end of drawdown

                return (self.short_trade_logs.loc[i,'Entry Time'] - self.short_trade_logs.loc[j,'Entry Time']).days
        
            self.performance_metrics.loc['Max Drawdown Duration', (self.metric_category+'--Short')] = max_drawdown_duration_calc(self)
            ###############################################

            def magic_number_calc(self):
                return round(((self.performance_metrics.loc['Profit Per Winning Trade', (self.metric_category+'--Short')]*self.performance_metrics.loc['Win Percentage', (self.metric_category+'--Short')]/100) + 
                             (self.performance_metrics.loc['Loss Per Losing Trade', (self.metric_category+'--Short')]*(1-(self.performance_metrics.loc['Win Percentage', (self.metric_category+'--Short')]/100)))), 2)
        
            self.performance_metrics.loc['Magic Number', (self.metric_category+'--Short')] = magic_number_calc(self)
            ################################################

            def monthly_perf_calc(self):
                return self.short_trade_logs.groupby(self.short_trade_logs['Entry Time'].dt.month)['PNL'].sum()
            
            self.monthly_performance[(self.metric_category+'--Short')] = monthly_perf_calc(self)
            ###############################################
            def yearly_perf_calc(self):
                return self.short_trade_logs.groupby(self.short_trade_logs['Entry Time'].dt.year)['PNL'].sum()
            
            self.yearly_performance[(self.metric_category+'--Short')] = yearly_perf_calc(self)
            ################################################
            def weekly_perf_calc(self):
                return self.short_trade_logs.groupby(self.short_trade_logs['Exit Time'].dt.dayofweek)['PNL'].sum()
            
            self.weekly_performance[(self.metric_category+'--Short')] = weekly_perf_calc(self)
            ################################################
            def hourly_entry_perf_calc(self):
                return self.short_trade_logs.loc[self.short_trade_logs['PNL']>0].groupby(self.short_trade_logs['Entry Time'].dt.hour)['PNL'].count()
            
            self.hourly_entry_performance[(self.metric_category+'--Short')] = hourly_entry_perf_calc(self)
        
            ################################################
            def hourly_exit_perf_calc(self):
                return self.short_trade_logs.groupby(self.short_trade_logs['Exit Time'].dt.hour)['PNL'].apply(np.mean)
            
            self.hourly_exit_performance[(self.metric_category+'--Short')] = hourly_exit_perf_calc(self)

            ################################################

    def plot_monthly_performance(self, calc_type='--Overall', path=None, save_plot=False):
            fig = px.bar( y=self.monthly_performance[self.metric_category+calc_type], x=self.monthly_performance.index, title='Monthly Performance',   width=1750, height=700)
            if save_plot:
                assert path is not None
                fig.write_html(path)
            fig.show()
        
    def plot_yearly_performance(self, calc_type='--Overall', path=None, save_plot=False):
            fig = px.bar( y=self.yearly_performance[self.metric_category+calc_type], x=self.yearly_performance.index, title='Yearly Performance',   width=1750, height=700)
            if save_plot:
                assert path is not None
                fig.write_html(path)
            fig.show()
        
    def plot_hourly_entry_performance(self, calc_type='--Overall', path=None, save_plot=False):
            fig = px.bar( y=self.hourly_entry_performance[self.metric_category+calc_type], x=self.hourly_entry_performance.index, title='Hourly Entry Performance',   width=1750, height=700)
            if save_plot:
                assert path is not None
                fig.write_html(path)
            fig.show()
        
    def plot_hourly_exit_performance(self, calc_type='--Overall', path=None, save_plot=False):
            fig = px.bar( y=self.hourly_exit_performance[self.metric_category+calc_type], x=self.hourly_exit_performance.index, title='Hourly Exit Performance',   width=1750, height=700)
            if save_plot:
                assert path is not None
                fig.write_html(path)
            fig.show()
        
    def plot_weekly_performance(self, calc_type='--Overall', path=None, save_plot=False):
            fig = px.bar(y=self.weekly_performance[self.metric_category+calc_type], x=self.weekly_performance.index, title='Weekly Performance',   width=1750, height=700)
            if save_plot:
                assert path is not None
                fig.write_html(path)
            fig.show()
        
    def plot_cumulative_returns(self, calc_type='--Overall', path=None, save_plot=False):
            fig = px.line( y= self.trade_logs['PNL'].cumsum(), x=self.trade_logs['Entry Time'], title='Cumulative Returns',   width=1750, height=700)
            if save_plot:
                assert path is not None
                fig.write_html(path)
            fig.show()
        
    def plot_daily_pnl(self, calc_type='--Overall', path=None, save_plot=False):
            df = pd.DataFrame(data = self.trade_logs[['PNL','Entry Time']])
            df["Color"] = np.where(df["PNL"]<0, 'red', 'green')    
            fig = go.Figure()
            fig.add_trace(go.Bar(name='',x=df['Entry Time'].dt.date,y=df['PNL'],marker_color=df['Color']))
            fig.update_layout(barmode='stack',   width=1750, height=700)
            if save_plot:
                assert path is not None
                fig.write_html(path)
            fig.show()
