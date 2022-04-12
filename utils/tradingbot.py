import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from utils.alpacaConnect import get_data 

# Signals & Features
def signals_features(ticker):
    btcusd_df= get_data(ticker)
    signals_df = btcusd_df.loc[:, ["close"]]
    signals_df["Actual Returns"] = signals_df["close"].pct_change()
    short_window = 50
    long_window = 100
    signals_df['SMA_Fast'] = signals_df['close'].rolling(window=short_window).mean()
    signals_df['SMA_Slow'] = signals_df['close'].rolling(window=long_window).mean()
    signals_df['Signal'] = 0.0
    signals_df.loc[(signals_df['Actual Returns'] >= 0), 'Signal'] = 1
    signals_df.loc[(signals_df['Actual Returns'] < 0), 'Signal'] = -1
    signals_df['Strategy Returns'] = signals_df['Actual Returns'] * signals_df['Signal'].shift()
    signals_df["Strategy Cummulative Returns"]= (1 + signals_df['Strategy Returns']).cumprod()
    signals_df = signals_df.dropna()

    return signals_df

# Mayer multiples: USE THIS FUNCTION FOR VISUALIZATION TESTS
def mayer_calculations(btcusd_df):

    mayer_window = 200  
    mayer_df = btcusd_df.loc[:, ["close"]].copy()
    mayer_df['SMA_200'] = btcusd_df['close'].rolling(window=mayer_window).mean()
    mayer_df['Mayer_Multiples'] = mayer_df['close'] / mayer_df['SMA_200']

    mayer_bands = mayer_df.loc[:, ["close"]].copy()
    mayer_bands = mayer_bands.drop(['SMA_200'], axis = 1)
    mayer_bands['Oversold'] = mayer_bands['Mayer_Multiples'] * 0.55
    mayer_bands['Bearish'] = mayer_bands['Mayer_Multiples'] * 1.1
    mayer_bands['Bullish'] = mayer_bands['Mayer_Multiples'] * 1.7
    mayer_bands['Bullish_Extension'] = mayer_bands['Mayer_Multiples'] * 2.5
   
    return mayer_df, mayer_bands

#### WORK ON IN CLASS and ask questions
def sharpe_visual(btcusd_df):
    sharpe_price_df = btcusd_df.loc[:, ["close"]].copy()
    sharpe_daily_returns = sharpe_price_df.pct_change().dropna() 
    stds = sharpe_daily_returns.std()
    annualized_stds = stds * np.sqrt(365)
    annualized_avg_returns = sharpe_daily_returns.mean() * 365
    sharpe_ratios = annualized_avg_returns / annualized_stds

    return sharpe_ratios 

#### USE THIS FUNCTION TO TEST VISUALS
def SMA_bands(btcusd_df):
    sma_df = btcusd_df.loc[:, ["close"]].copy()
    window_10 = 10
    window_20 = 20
    window_50 = 50
    window_100 = 100
    window_200 = 200
    sma_df['SMA_10'] = sma_df['close'].rolling(window=window_10).mean()
    sma_df['SMA_20'] = sma_df['close'].rolling(window=window_20).mean()
    sma_df['SMA_50'] = sma_df['close'].rolling(window=window_50).mean()
    sma_df['SMA_100'] = sma_df['close'].rolling(window=window_100).mean()
    sma_df['SMA_200'] = sma_df['close'].rolling(window=window_200).mean()

    return sma_df 

#### ASK QUESTION ABOUT DATE ORDER
def SMA_1458(btcusd_df):
    sma_1458_df = btcusd_df.loc[:, ["close"]].copy()
    window_1458 = 1458
    sma_1458_df['SMA_1458'] = sma_1458_df['close'].rolling(window=window_1458).mean() 

    return sma_1458_df


def  golden_ratio_multiplier(btcusd_df): 
    ma_golden_df = btcusd_df.loc[:, ["close"]].copy() 
    ma_golden = 350 
    ma_golden_df['MA_350'] =  ma_golden_df['close'].rolling(window=ma_golden).mean()
    ma_golden_df['Golden_Ratio_Multiplier'] = ma_golden_df['MA_350'] * 1.6
    ma_golden_df['2'] = ma_golden_df['MA_350'] * 2
    ma_golden_df['3'] = ma_golden_df['MA_350'] * 3
    ma_golden_df['5'] = ma_golden_df['MA_350'] * 5
    ma_golden_df['8'] = ma_golden_df['MA_350'] * 8
    ma_golden_df['13'] = ma_golden_df['MA_350'] * 13

    return ma_golden_df 