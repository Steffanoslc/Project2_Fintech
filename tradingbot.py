import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from utils.alpacaConnect import get_data 

<<<<<<< HEAD:utils/tradingbot.py
# Signals & Features
def signals_features(ticker):
    btcusd_df= get_data(ticker)
    signals_df = btcusd_df.loc[:, ["close"]]
=======


###### CONDENSE INTO A FUNCTION BEFORE SATURDAY OFFICE HOURS

#### TEST/WORK ON IN CLASS MONDAY
def tradingbot():



    # Filter the date index and close columns
    signals_df = btcusd_df.copy()

    # Use the pct_change function to generate  returns from close prices
>>>>>>> 61ddde5ba92146413cf4c3a4e1d02b20ca21a210:tradingbot.py
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

<<<<<<< HEAD:utils/tradingbot.py
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
=======
    mayer_window = 200 

   
    mayer_df = btcusd_df.copy()
    mayer_df['SMA_200'] = btcusd_df['4b. close (USD)'].rolling(window=mayer_window).mean()

    mayer_df['Mayer_Multiples'] = mayer_df['4b. close (USD)'] / mayer_df['SMA_200']



    close_price = mayer_df[["4b. close (USD)"]].hvplot(
    line_color='lightgray',
    ylabel='Price in $',
    width=1000,
    height=400
    )

    multiples = mayer_df[['Mayer_Multiples']].hvplot(
    ylabel='Multiple',
    width=1000,
    height=400
    )

    mayer_plot = close_price * multiples
    mayer_plot


    ### mayer multiple bands
    mayer_bands = btcusd_df.loc[:, ["4b. close (USD)"]].copy()
    mayer_bands['Multiples'] = mayer_df['Mayer_Multiples'] 

    mayer_bands['Oversold'] = mayer_bands['Mayer_Multiples'] * 0.55 * mayer_bands['4b. close (USD)']
    mayer_bands['Bearish'] = mayer_bands['Mayer_Multiples'] * 1.1 * mayer_bands['4b. close (USD)']
    mayer_bands['Bullish'] = mayer_bands['Mayer_Multiples'] * 1.7 * mayer_bands['4b. close (USD)']
    mayer_bands['Bullish_Extension'] = mayer_bands['Mayer_Multiples'] * 2.5 * mayer_bands['4b. close (USD)']
>>>>>>> 61ddde5ba92146413cf4c3a4e1d02b20ca21a210:tradingbot.py
   
    return mayer_df, mayer_bands

#### WORK ON IN CLASS and ask questions
def sharpe_visual(btcusd_df):
    sharpe_price_df = btcusd_df.copy()
    sharpe_change_df = sharpe_price_df    
    sharpe_daily_returns = sharpe_change_df.pct_change().dropna() 
    sharpe_daily_returns.columns = ['pct_change']
    stds = sharpe_daily_returns['pct_change'].std()
    sharpe_ratios= (sharpe_daily_returns - 0.0369) / stds
    sharpe_ratios.columns = ['Sharpe Ratios']
    ratio_plot = sharpe_ratios['Sharpe Ratios'].hvplot(ylim = [-5,6])
    price_plot = sharpe_price_df['close'].hvplot()

    sharpe_plot = ratio_plot * price_plot 

    return sharpe_plot 

#### USE THIS FUNCTION TO TEST VISUALS
def SMA_bands(btcusd_df):
<<<<<<< HEAD:utils/tradingbot.py
    sma_df = btcusd_df.loc[:, ["close"]].copy()
=======
    sma_df = btcusd_df.copy()

>>>>>>> 61ddde5ba92146413cf4c3a4e1d02b20ca21a210:tradingbot.py
    window_10 = 10
    window_20 = 20
    window_50 = 50
    window_100 = 100
    window_200 = 200
<<<<<<< HEAD:utils/tradingbot.py
    sma_df['SMA_10'] = sma_df['close'].rolling(window=window_10).mean()
    sma_df['SMA_20'] = sma_df['close'].rolling(window=window_20).mean()
    sma_df['SMA_50'] = sma_df['close'].rolling(window=window_50).mean()
    sma_df['SMA_100'] = sma_df['close'].rolling(window=window_100).mean()
    sma_df['SMA_200'] = sma_df['close'].rolling(window=window_200).mean()

    return sma_df 
=======

    sma_df['SMA_10'] = sma_df['4b. close (USD)'].rolling(window=window_10).mean()

    sma_df['SMA_20'] = sma_df['4b. close (USD)'].rolling(window=window_20).mean()

    sma_df['SMA_50'] = sma_df['4b. close (USD)'].rolling(window=window_50).mean()

    sma_df['SMA_100'] = sma_df['4b. close (USD)'].rolling(window=window_100).mean()

    sma_df['SMA_200'] = sma_df['4b. close (USD)'].rolling(window=window_200).mean()

    close_price_sma = sma_df[["4b. close (USD)"]].hvplot(
    line_color='lightgray',
    ylabel='Price in $',
    width=1000,
    height=400
    )

    sma_10 = sma_df[['SMA_10']].hvplot(
    width=1000,
    height=400
    )

    sma_20 = sma_df[['SMA_20']].hvplot(
    width=1000,
    height=400
    )

    sma_50 = sma_df[['SMA_50']].hvplot(
    width=1000,
    height=400
    )

    sma_100 = sma_df[['SMA_100']].hvplot(
    width=1000,
    height=400
    )

    sma_200 = sma_df[['SMA_200']].hvplot(
    width=1000,
    height=400
    )

    sma_plot = close_price_sma * sma_10 * sma_20 * sma_50 * sma_100 * sma_200 
    sma_plot 

    return sma_plot 
>>>>>>> 61ddde5ba92146413cf4c3a4e1d02b20ca21a210:tradingbot.py

#### ASK QUESTION ABOUT DATE ORDER
def SMA_1458(btcusd_df):
    sma_1458_df = btcusd_df.loc[:, ["close"]].copy()
    window_1458 = 1458
    sma_1458_df['SMA_1458'] = sma_1458_df['close'].rolling(window=window_1458).mean() 

    return sma_1458_df


<<<<<<< HEAD:utils/tradingbot.py
def  golden_ratio_multiplier(btcusd_df): 
    ma_golden_df = btcusd_df.loc[:, ["close"]].copy() 
=======
    sma_1458_plot = close_price_sma_1458 * sma_1458 
    sma_1458_plot
    
    return sma_1458_plot

##### ASK QUESTIONS MONDAY 
def  golden_ratio_multiplier(btcusd_df): 
    ma_golden_df = btcusd_df.copy() 

>>>>>>> 61ddde5ba92146413cf4c3a4e1d02b20ca21a210:tradingbot.py
    ma_golden = 350 
    ma_golden_df['MA_350'] =  ma_golden_df['close'].rolling(window=ma_golden).mean()
    ma_golden_df['Golden_Ratio_Multiplier'] = ma_golden_df['MA_350'] * 1.6
    ma_golden_df['2'] = ma_golden_df['MA_350'] * 2
    ma_golden_df['3'] = ma_golden_df['MA_350'] * 3
    ma_golden_df['5'] = ma_golden_df['MA_350'] * 5
    ma_golden_df['8'] = ma_golden_df['MA_350'] * 8
    ma_golden_df['13'] = ma_golden_df['MA_350'] * 13

    return ma_golden_df 