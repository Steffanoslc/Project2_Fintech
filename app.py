
import pandas as pd
from sqlalchemy import all_
from sympy import viete
from utils.tradingbot import signals_features
from utils.svmmodel import svmmodel
import streamlit as st
from utils.alpacaConnect import get_data
from utils.tradingbot import SMA_bands
from utils.tradingbot import SMA_1458
from utils.tradingbot import golden_ratio_multiplier
from utils.twitterConnect import twitter_df


ticker= "BTCUSD"
url= "https://api.alternative.me/fng/?"
search_term="#bitcoin -filter:retweets"




def run():
    signals_df= signals_features(ticker)
    predictions= svmmodel(signals_df)
    
    #Trading bot plot
    st.title('Team Five Project')
    st.header('Bitcoin Machine Learning Model')
    st.subheader('In this project, we built a machine learning model to trade bitcoin. We pulled data using the Alapaca API and calculated moving averages and certain bitcoin-specific indicators such as meyer multiples. We passed the dataframe into an SVM model from sklearn and results are presented on this page using streamlit.')    
    st.header('Cumulative Performance Line Chart')
    st.line_chart(predictions[["Commulative Actual Returns", "Commulative Trading Algorithm Returns"]])
    st.header('Predictions Dataframe')
    st.table(predictions.head(20))

    #SMA Bands plot
    btcusd_df= get_data(ticker)
    SMAbands_df= SMA_bands(btcusd_df)
    st.header('SMA Bands')
    st.line_chart(SMAbands_df)

    #SMA 1458 plot
    btcusd_df= get_data(ticker)
    SMA1458_df= SMA_1458(btcusd_df)
    st.header('SMA 1458')
    st.line_chart(SMA1458_df)

    #Golden Ratio Multiplier plot
    btcusd_df= get_data(ticker)
    GRM_df= golden_ratio_multiplier(btcusd_df)
    st.header('Golden Ratio Multiplier')
    st.line_chart(GRM_df)

    #Twittwer Sentiment plot
    sentiment_df= twitter_df(search_term)
    st.header('Twitter scatter plot')
    st.table(sentiment_df)

if __name__ == "__main__":
    run()