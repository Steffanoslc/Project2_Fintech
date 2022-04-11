
import pandas as pd
from utils.tradingbot import signals_features
from utils.svmmodel import svmmodel


ticker= "BTCUSD"
url= "https://api.alternative.me/fng/?"








def run():
    signals_df= signals_features(ticker)
    predictions= svmmodel(signals_df)
    print(predictions)



if __name__ == "__main__":
    run()