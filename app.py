
import pandas as pd
from utils.tradingbot import signals_features
from utils.svmmodel import svmmodel
import streamlit as st

ticker= "BTCUSD"
url= "https://api.alternative.me/fng/?"








def run():
    signals_df= signals_features(ticker)
    predictions= svmmodel(signals_df)
    print(predictions)
    st.title('Team Five Project')
    st.line_chart(predictions)



if __name__ == "__main__":
    run()