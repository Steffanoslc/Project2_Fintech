# Project2_Fintech

The app uses a modularize approach to pull 8 years of closing prices for BTC using Alpaca API, search for the last 300 tweets using the twitter API, and the last 3 years of the Fear & Greed index using the Alternative.me API.

With all the data a set of technical analysis are generated. The combination of technical analysis and sentiment is pass in into a machine learning model (SVM) and trained with it.

Lastly, we plot our result in Streamlit.io

https://docs.google.com/presentation/d/1Hm6LZr3PxRha1Opb0oz1ahUR8RVjo7jHYcdIGm1mJBA/edit#slide=id.gc6f9e470d_0_37