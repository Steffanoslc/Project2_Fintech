import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset

def svmmodel(signals_df):
    X = signals_df[['SMA_Fast', 'SMA_Slow']].shift().dropna()
    y = signals_df['Signal']
    training_begin = X.index.min()
    training_end = X.index.min() + DateOffset(months=3)
    X_train = X.loc[training_begin:training_end]
    y_train = y.loc[training_begin:training_end]
    X_test = X.loc[training_end+DateOffset(hours=1):]
    y_test = y.loc[training_end+DateOffset(hours=1):]
    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    svm_model = svm.SVC()
    svm_model = svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)
    predictions_df = pd.DataFrame(index=X_test.index)
    predictions_df['Predicted'] = svm_pred
    predictions_df['Actual Returns'] = signals_df['Actual Returns']
    predictions_df['trading_algorithm_returns'] = predictions_df["Actual Returns"] * predictions_df["Predicted"]
    predictions_df["Commulative Actual Returns"]= (1 + predictions_df['Actual Returns']).cumprod()
    predictions_df["Commulative Trading Algorithm Returns"]= (1 - predictions_df['trading_algorithm_returns']).cumprod()

    return predictions_df

