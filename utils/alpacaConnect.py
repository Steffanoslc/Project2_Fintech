import os
import datetime
import pandas as pd
from datetime import date
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame

def get_data(ticker):
    load_dotenv()
    alpaca_api_key= os.getenv("ALPACA_API_KEY")
    alpaca_secret_key= os.getenv("ALPACA_SECRET_KEY")
    alpaca = REST(
        alpaca_api_key,
        alpaca_secret_key,
        api_version="v2"
    )
    start_date= date.today().strftime("%Y-%m-%d")
    end_date= date.today() - datetime.timedelta(days=8*365)
    data = alpaca.get_crypto_bars(ticker, TimeFrame.Day, end_date, start_date).df
    btcusd_df= data.drop(columns=["open","high","low","volume","trade_count","vwap"])
    btcusd_df= btcusd_df.loc[btcusd_df["exchange"]== "CBSE"]
    btcusd_df= btcusd_df.drop(columns="exchange")
    return btcusd_df
