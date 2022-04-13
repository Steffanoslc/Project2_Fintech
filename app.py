
import pandas as pd
from sympy import viete
from utils.streamlit import visualization
from utils.tradingbot import signals_features
from utils.svmmodel import svmmodel


ticker= "BTCUSD"
url= "https://api.alternative.me/fng/?"





def run():
    signals_df= signals_features(ticker)
    predictions= svmmodel(signals_df)
    streamlit_visualizations= visualization(predictions)
    return streamlit_visualizations



if __name__ == "__main__":
    run()