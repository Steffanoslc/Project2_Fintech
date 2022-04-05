import os
import datetime
import pandas as pd
from datetime import date
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

ticker= "BTCUSD"

load_dotenv()
alpaca_api_key= os.getenv("ALPACA_API_KEY")
alpaca_secret_key= os.getenv("ALPACA_SECRET_KEY")

alpaca = tradeapi.REST(
    alpaca_api_key,
    alpaca_secret_key,
    api_version="v2"
)
today= date.today().strftime("%Y-%m-%d")
year_3= date.today() - datetime.timedelta(days=3*365)

start_date = pd.Timestamp(year_3).isoformat()
end_date = pd.Timestamp(today).isoformat()
timeframe = "1D"
limit_rows = 1000


# loop through tickers and get data 

"""
def get_data(ticker):
    data = alpaca.get_barset(
        ticker,
        timeframe,
        start= start_date,
        end= end_date,
        limit= limit_rows
    ).df
    return data

"""
btcusd_df= alpaca.get_crypto_quotes_iter(
    ticker,
    timeframe,
    start= start_date,
    end= end_date,
    limit= limit_rows
).df

print(btcusd_df)

