import pandas as pd
import requests
import json

class Data():
    def getPrice(ticker):
        #Get stock data for a ticker symbol
        #url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + ticker + '&outputsize=full&apikey=7ZET74D05LNJ0FOF'
        url =  'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&outputsize=full&apikey=demo'
        response = requests.get(url)

        if response.status_code == 200:
        #Parse data and convert to DataFrame
            data = json.loads(response.text) 
            time_series = data['Time Series (Daily)']
            stock_data = []

            for date, values in time_series.items():
                stock_data_point = {
                    'date': date,
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                }
                stock_data.append(stock_data_point)

            stock_df = pd.DataFrame(stock_data)
            stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date
            stock_df.set_index('date', inplace=True)
            stock_df.sort_index(inplace=True)
            return stock_df
        else:
            print("Failed to get data")
            return None

class SI():
    def EMA(stock_df, window):
        if not(window):
            window = 10
        # Calculate the EMA using the pandas ewm function
        stock_df['EMA'] = stock_df['close'].ewm(span=window).mean()
        return stock_df

    def SMA(stock_df, window):
        # Calculate the SMA using the pandas rolling function
        stock_df['SMA'] = stock_df['close'].rolling(window=window).mean()
        return stock_df

    def RSI(stock_df, window):
        # Calculate the RSI using the pandas rolling function
        delta = stock_df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Prevent division by zero
        loss_plus_1 = loss + 1

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        stock_df['RSI'] = rsi
        return stock_df
    
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions

class sentimentAnalysis():

    def analyze_sentiment(stock_df):
        api_key = 'your_ibm_watson_api_key'
        url = 'https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/your_instance_id'

        authenticator = IAMAuthenticator(api_key)
        natural_language_understanding = NaturalLanguageUnderstandingV1
        version='2021-08-01',
        authenticator=authenticator


        natural_language_understanding.set_service_url(url)

        sentiment_score = []

        for price in stock_df:
            try:
                sentiment = natural_language_understanding.analyze(
                    text=str(price),
                    features=Features(sentiment=SentimentOptions())).get_result()
                sentiment_score.append(sentiment['sentiment']['document']['score'])
            except:
                sentiment_score.append(0)
        
        stock_df['sentiment'] = sentiment_score
        return stock_df

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class StockPricePredictor:

    def __init__(self, n_timesteps, n_features, n_epochs=100, batch_size=32, learning_rate=0.001):
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        model = nn.LSTM(input_size=self.n_features, hidden_size=50, num_layers=1, batch_first=True)
        model = model.double()
        return model

    def fit(self, data):
        # Create the input-output dataset
        data = data.values.astype('float32')
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data = scaler.fit_transform(data)
        x = np.array([data[i:i + self.n_timesteps] for i in range(len(data) - self.n_timesteps)])
        y = np.array([data[i + self.n_timesteps] for i in range(len(data) - self.n_timesteps)])
        x = np.reshape(x, (x.shape[0], 1, x.shape[1]))

        # Create the dataloader
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Define the loss and the optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Train the model
        for epoch in range(self.n_epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.double(), labels.double()
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def predict(self, data):
        data = data.values.astype('float32')
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data = scaler.fit_transform(data)
        x = np.array([data[i:i + self.n_timesteps] for i in range(len(data) - self.n_timesteps)])
        x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
        x = torch.from_numpy(x).double()
        outputs = self.model(x)
        return outputs.detach().numpy()

stock_data = pd.read_csv('your_stock_data.csv')
stock_data = stock_data.dropna()

# Create a train-test split
train_size = int(len(stock_data) * 0.8)
train_data, test_data = stock_data[0:train_size], stock_data[train_size:]

model = StockPricePredictor(n_timesteps=50, n_features=1)
model.fit(train_data)
predictions = model.predict(test_data)

# Inverse transform the predictions to obtain the original scale
predictions = scaler.inverse_transform(predictions)
test_data = scaler.inverse_transform(test_data.values)

# Calculate the root mean squared error
rmse = np.sqrt(mean_squared_error(test_data, predictions))
print('Root Mean Squared Error:', rmse)