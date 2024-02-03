import matplotlib as plt
import pandas as pd
import numpy as np

import keras
from keras import layers
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from keras.layers import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

stock_data = pd.read_csv(r"C:\Users\brand\Desktop\NVDA.csv", index_col='Date')
stock_data.head()

target_y = stock_data['Close']
X_feat = stock_data.iloc[:,0:3]

sc = StandardScaler()
X_ft = sc.fit_transform(X_feat.values)
X_ft = pd.DataFrame(columns=X_feat.columns,
                    data = X_ft,
                    index = X_feat.index)

def lstm_split(data, n_steps):
    X, Y = [], []
    for i in range (len(data)-n_steps+1):
        X.append(data[i:i + n_steps, :-1])
        Y.append(data[i + n_steps - 1, -1])
    return np.array(X), np.array(Y)

X1, Y1 = lstm_split(stock_data.values, n_steps=2)

train_split = 0.8
split_idx = int(np.ceil(len(X1) * train_split))
date_index = stock_data.index

x_train, x_test = X1[:split_idx], X1[split_idx:]
y_train, y_test = Y1[:split_idx], Y1[split_idx:]
x_train_date, x_test_date = date_index[:split_idx], date_index[split_idx]

lstm = Sequential()
lstm.add(LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2]),
              activation='relu', return_sequences=True))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer= 'adam')

histroy = lstm.fit(x_train, y_train,
                   epochs=100,batch_size=4,
                   verbose=2, shuffle=False)

y_pred = lstm.predict(x_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mape = mean_absolute_percentage_error(y_test, y_pred)

lstm = Sequential()
lstm.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2]),
              activation='relu', return_sequences=True))
lstm.add(LSTM(50, activation='relu'))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')

rmse = mean_squared_error(y_test, y_pred, squared=False)
mape = mean_absolute_percentage_error(y_test, y_pred)

n_steps = 10
X1, y1 = lstm_split(stock_data.values, n_steps=n_steps)

train_split=0.8
split_idx = int(np.ceil(len(X1) * train_split))
date_index = stock_data.index

x_train, x_test = X1[:split_idx], X1[split_idx:]
y_train, y_text = y1[:split_idx], y1[split_idx:]
x_train_date, x_train_date = date_index[:split_idx], date_index[split_idx: -n_steps]

rmse = mean_squared_error(y_test, y_pred, squared=False)
mape = mean_absolute_percentage_error(y_test, y_pred)

train_split = 0.8
split_idx = int(np.ceil(len(stock_data)*train_split))
train = stock_data[['Close']].iloc[:split_idx]
test = stock_data[['Close']].iloc[split_idx:]

test_pred = np.arry([train.rolling(10).mean().iloc[-1]] * len(test)).reshape((-1,1))

print('Test RMSE: %.3f' % mean_squared_error(test, test_pred, squared=False))
print('Test Mape: %.3f' % mean_absolute_percentage_error(test, test_pred))

plt.figure(figsize=(10,5))
plt.plot(test)
plt.plot(test_pred)
plt.show()

X = stock_data[['Close']].values
train_split = 0.8
split_idx = int(np.ceil(len(X)*train_split))
train = X[:split_idx]
test = X[split_idx:]
test_concat = np.array([]).reshape((0,1))

for i in range(len(test)):
    train_fit = np.concatenate((train, np.asarray(test_concat)))
    test_pred = train_fit.forcast(1)
    test_concat = np.concatenate((np.asarray(test_concat),test_pred.reshape((-1,1))))

print('Test RMSE: %.3f') % mean_squared_error(test, test_concat, squared=False)
print('Test MAPE: %.3f') % mean_absolute_percentage_error(test, test_concat)