
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
    st.header('Bitcoin Machine Learning Model')
    st.subheader('In this project, we built a machine learning model to trade bitcoin. We pulled data using the Alapaca API and calculated moving averages and certain bitcoin-specific indicators such as meyer multiples. We passed the dataframe into an SVM model from sklearn and results are presented on this page using streamlit.')
    
       
    st.header('Cumulative Performance Line Chart')
    st.line_chart(predictions)


    st.header('Predictions Dataframe')
    st.table(predictions.head(20))



if __name__ == "__main__":
    run()