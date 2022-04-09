import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report
from utils.alpacaConnect import btcusd_df 
#import csv



###### CONDENSE INTO A FUNCTION BEFORE SATURDAY OFFICE HOURS
def tradingbot():



    # Filter the date index and close columns
    signals_df = btcusd_df.loc[:, ["close"]]

    # Use the pct_change function to generate  returns from close prices
    signals_df["Actual Returns"] = signals_df["close"].pct_change()

    # Drop all NaN values from the DataFrame
    signals_df = signals_df.dropna()

    # Review the DataFrame
    display(signals_df.head())
    display(signals_df.tail())

    # Set the short window and long window
    short_window = 50
    long_window = 100

    # Generate the fast and slow simple moving averages (4 and 100 days, respectively)
    signals_df['SMA_Fast'] = signals_df['close'].rolling(window=short_window).mean()
    signals_df['SMA_Slow'] = signals_df['close'].rolling(window=long_window).mean()

    signals_df = signals_df.dropna()

    # Review the DataFrame
    display(signals_df.head())
    display(signals_df.tail())

    # Initialize the new Signal column
    signals_df['Signal'] = 0.0

    # When Actual Returns are greater than or equal to 0, generate signal to buy stock long
    signals_df.loc[(signals_df['Actual Returns'] >= 0), 'Signal'] = 1

    # When Actual Returns are less than 0, generate signal to sell stock short
    signals_df.loc[(signals_df['Actual Returns'] < 0), 'Signal'] = -1

    # Review the DataFrame
    display(signals_df.head())
    display(signals_df.tail())

    signals_df['Signal'].value_counts()

    # Calculate the strategy returns and add them to the signals_df DataFrame
    signals_df['Strategy Returns'] = signals_df['Actual Returns'] * signals_df['Signal'].shift()

    # Review the DataFrame
    display(signals_df.head())
    display(signals_df.tail())

    # Plot Strategy Returns to examine performance
    (1 + signals_df['Strategy Returns']).cumprod().plot()

    # Assign a copy of the sma_fast and sma_slow columns to a features DataFrame called X
    X = signals_df[['SMA_Fast', 'SMA_Slow']].shift().dropna()

    # Review the DataFrame
    X.head()

    # Create the target set selecting the Signal column and assiging it to y
    y = signals_df['Signal']

    # Review the value counts
    y.value_counts()

    # Select the start of the training period
    training_begin = X.index.min()

    # Display the training begin date
    print(training_begin)

    # Select the ending period for the training data with an offset of 3 months
    training_end = X.index.min() + DateOffset(months=3)

    # Display the training end date
    print(training_end)

    # Generate the X_train and y_train DataFrames
    X_train = X.loc[training_begin:training_end]
    y_train = y.loc[training_begin:training_end]

    # Review the X_train DataFrame
    X_train.head()

    # Generate the X_test and y_test DataFrames
    X_test = X.loc[training_end+DateOffset(hours=1):]
    y_test = y.loc[training_end+DateOffset(hours=1):]

    # Review the X_test DataFrame
    X_train.head()

    # Scale the features DataFrames

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Apply the scaler model to fit the X-train data
    X_scaler = scaler.fit(X_train)

    # Transform the X_train and X_test DataFrames using the X_scaler
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    # From SVM, instantiate SVC classifier model instance
    svm_model = svm.SVC()
    
    # Fit the model to the data using the training data
    svm_model = svm_model.fit(X_train_scaled, y_train)
    
    # Use the testing data to make the model predictions
    svm_pred = svm_model.predict(X_test_scaled)

    # Review the model's predicted values
    svm_pred[:10]

    # Use a classification report to evaluate the model using the predictions and testing data
    svm_testing_report = classification_report(y_test, svm_pred)

    # Print the classification report
    print(svm_testing_report)

    # Create a new empty predictions DataFrame:

    # Create a predictions DataFrame
    predictions_df = pd.DataFrame(index=X_test.index)

    # Add the SVM model predictions to the DataFrame
    predictions_df['Predicted'] = svm_pred

    # Add the actual returns to the DataFrame
    predictions_df['Actual Returns'] = signals_df['Actual Returns']

    # Add the strategy returns to the DataFrame
    predictions_df['Strategy Returns'] = signals_df['Strategy Returns']

    # Review the DataFrame
    display(predictions_df.head())
    display(predictions_df.tail())

    #plot cumulative returns
    (1 + predictions_df[['Actual Returns', 'Strategy Returns']]).cumprod().hvplot(
        title='Baseline Actual v. Strategy Returns with SMA 50 & 100'
        )

    # Import a new classifier from SKLearn
    from sklearn.ensemble import RandomForestClassifier

    # Initiate the model instance
    ml_model = RandomForestClassifier()

    # Fit the model using the training data
    ml_model.fit(X_train_scaled, y_train)

    # Use the testing dataset to generate the predictions for the new model
    ml_pred = ml_model.predict(X_test_scaled)

    # Review the model's predicted values
    ml_pred[:10]

    # Use a classification report to evaluate the model using the predictions and testing data
    ml_testing_report = classification_report(y_test, ml_pred)

    # Print the classification report
    print(ml_testing_report)

    # Create a new empty predictions DataFrame:

    # Create a predictions DataFrame
    ml_predictions_df = pd.DataFrame(index=X_test.index)

    # Add the Logistic Regression model predictions to the DataFrame
    ml_predictions_df['Strategy Returns'] = ml_pred

    # Add the actual returns to the DataFrame
    ml_predictions_df['Actual Returns'] = signals_df['Actual Returns']

    # Add the strategy returns to the DataFrame
    ml_predictions_df['Trading Algorithm Returns'] = (
        ml_predictions_df['Actual Returns'] * ml_predictions_df['Strategy Returns']
    )

    ml_predictions_df = ml_predictions_df.loc['2018-06-25':'2021-06-25']
    # Review the DataFrame
    display(ml_predictions_df.head())
    display(ml_predictions_df.tail())
    # Plot the actual returns versus the strategy returns
    (1 + ml_predictions_df[['Actual Returns', 'Trading Algorithm Returns']]).cumprod().hvplot(
        title='Random Forest Actual v. Strategy Returns with SMA 50 & 100: 36 Month Window'
        )

#### Define returns and what output is needed

# Mayer multiples
def mayer_calculations(btcusd_df):

    mayer_window = 200 

   
    mayer_df = btcusd_df.loc[:, ["close"]].copy()
    mayer_df['SMA_200'] = btcusd_df['close'].rolling(window=mayer_window).mean()

    mayer_df['Mayer_Multiples'] = mayer_df['close'] / mayer_df['SMA_200']

    print(mayer_df)

    close_price = mayer_df[["close"]].hvplot(
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
    mayer_bands = mayer_df.loc[:, ["close"]].copy()

    mayer_bands = mayer_bands.drop(['SMA_200'], axis = 1)

    mayer_bands['Oversold'] = mayer_bands['Mayer_Multiples'] * 0.55
    mayer_bands['Bearish'] = mayer_bands['Mayer_Multiples'] * 1.1
    mayer_bands['Bullish'] = mayer_bands['Mayer_Multiples'] * 1.7
    mayer_bands['Bullish_Extension'] = mayer_bands['Mayer_Multiples'] * 2.5
   
    multiple_1 = mayer_bands[['Oversold']].hvplot(
    ylabel='Oversold',
    width=1000,
    height=400
    )

    multiple_2 = mayer_bands[['Bearish']].hvplot(
    ylabel='Bearish',
    width=1000,
    height=400
    )

    multiple_3 = mayer_bands[['Bullish']].hvplot(
    ylabel='Bullish',
    width=1000,
    height=400
    )

    multiple_4 = mayer_bands[['Bullish_Extension']].hvplot(
    ylabel='Bullish_Extension',
    width=1000,
    height=400
    )

    band_plot = close_price * multiple_1 * multiple_2 * multiple_3 * multiple_4
    band_plot

    return band_plot, mayer_plot


def sharpe_visual(btcusd_df):
   
    sharpe_price_df = btcusd_df.loc[:, ["close"]].copy()
    
    sharpe_daily_returns = sharpe_price_df.pct_change().dropna() 

    stds = sharpe_daily_returns.std()

    annualized_stds = stds * np.sqrt(365)

    annualized_avg_returns = sharpe_daily_returns.mean() * 365

    sharpe_ratios = annualized_avg_returns / annualized_stds

    print(sharpe_ratios)

    btc_price_chart = btcusd_df.hvplot(
    line_color='lightgray',
    ylabel='Price in $',
    width=1000,
    height=400
    )

    sharpe_chart = sharpe_ratios.hvplot(
    ylabel='Sharpe Ratios',
    width=1000,
    height=400
    )


    sharpe_plot = btc_price_chart * sharpe_chart 
    sharpe_plot

    return sharpe_plot 
    

