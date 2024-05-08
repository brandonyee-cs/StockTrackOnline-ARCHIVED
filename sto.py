import time
import config
import json
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class df:
    def __init__(self, ticker) -> None:
        self.stock_df = df.getData(ticker)
        self.ticker = ticker
        self.newskey = config.newskey
        self.vantagekey = config.vantagekey

    def getData(ticker): #Get stock data for a ticker symbol
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + ticker + '&outputsize=full&apikey=7ZET74D05LNJ0FOF'
        response = requests.get(url) #url =  'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&outputsize=full&apikey=demo'
        if response.status_code == 200:
            data = json.loads(response.text)
            time_series = data['Time Series (Daily)']
            stock_data = [] #Parse data and convert to DataFrame
            for date, values in time_series.items(): 
                stock_data_point = {'date': date, 
                                    'open': float(values['1. open']), 
                                    'high': float(values['2. high']), 
                                    'low': float(values['3. low']), 
                                    'close': float(values['4. close']), 
                                    'volume': int(values['5. volume'])}
                stock_data.append(stock_data_point)
            stock_df = pd.DataFrame(stock_data)
            stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date
            stock_df.set_index('date', inplace=True)
            stock_df.sort_index(inplace=True)
            return stock_df
        else: print("Failed to get data"); return None

class stockAnalysis(df):
    def __init__(self, window) -> None:
        super().__init__()
        self.window = window

    def EMA(self):

        if not(window): window = 10
        self.stock_df['EMA'] = self.stock_df['close'].ewm(span=self.window).mean()
        return self.stock_df #Calculate the EMA using the pandas ewm function

    def SMA(self): 

        self.stock_df['SMA'] = self.stock_df['close'].rolling(window=self.window).mean()
        return self.stock_df #Calculate the SMA using the pandas rolling function

    def RSI(self): 

        delta = self.stock_df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0) #Calculate the RSI using the pandas rolling function

        loss_plus_1 = loss + 1
        avg_gain = gain.rolling(window=self.window).mean()
        avg_loss = loss.rolling(window=self.window).mean()
        rs = avg_gain / avg_loss; rsi = 100 - (100 / (1 + rs))
        self.stock_df['RSI'] = rsi
        
        return self.stock_df #Prevent division by zero
    
    def BOLLINGER_BANDS(self):

        if not(window) or not(k): window = 20; k = 2
        self.stock_df['SMA'] = self.stock_df['close'].rolling(window=window).mean()
        self.stock_df['RSD'] = self.stock_df['close'].rolling(window=window).std()
        self.stock_df['UBB'] = self.stock_df['SMA'] + k * self.stock_df['RSD']
        self.stock_df['LBB'] = self.stock_df['SMA'] - k * self.stock_df['RSD']
        
        return self.stock_df #Calculate the rolling mean (SMA), rolling standard deviation (RSD), Upper Bollinger Band (UBB), and Lower Bollinger Band (LBB)

class publicSentiment(df):
    def __init__(self) -> None:
        super().__init__()
        self.interval = 50

    def SA(self):
        api_url = f'https://www.alphavantage.co/query?function=daily&symbol={self.ticker}&market=US&interval={self.interval}&apikey={self.vantagekey}'
        response = requests.get(api_url)
        data = response.json() #Send the GET request and parse the response
        sentiment_data = data['Time Series Sentiment Index'] #Extract the sentiment data
        for date, sentiment in sentiment_data.items(): print(f'Date: {date}, Sentiment Score: {sentiment["sentiment_score"]}')

    def get_news(self):
        api_url = f'https://newsapi.org/v2/everything?q={self.ticker}&apiKey={self.newskey}'
        response = requests.get(api_url)
        articles = response.json()['articles']
        for article in articles:
            print(f'Title: {article["title"]}, URL: {article["url"]}')



class LSTM(nn.Module):
    def initialize(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm((self.stock_df).view(len(self.stock_df) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(self.stock_df), -1))
        self.predictions = predictions[-1]

    def processdata(self):
        # Load and preprocess the data
        stock_data = pd.read_csv('stock_data.csv'); stock_data = stock_data['Close'].values; stock_data = stock_data.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1)); stock_data = scaler.fit_transform(stock_data)

        # Split the data into training and testing sets
        train_split = 0.8
        split_idx = int(len(stock_data) * train_split)
        train = stock_data[:split_idx]
        train_window = 50
        test = stock_data[split_idx:]

        model = LSTM()
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        epochs = 150

        for i in range(epochs):
            for seq, labels in train:
                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))

                y_pred = model(seq)

                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

            if i%25 == 1:
                print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

        # Test the model
        fut_pred = 12

        test_inputs = train[-train_window:].tolist()
        model.eval()

        for i in range(fut_pred):
            seq = torch.FloatTensor(test_inputs[-train_window:])
            with torch.no_grad():
                model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))
                test_inputs.append(model(seq).item())

        actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))
        self.test = test
        self.predictions = actual_predictions

class graphShow(LSTM):
    def __init__(self):
        super().__init__()

    def plot_close(self):
        plt.figure(figsize=(14,7))
        plt.plot(self.stock_df['close'])
        plt.title('Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()

    def plot_predictions(self):
        plt.plot(self.test, color='red',label='Real Stock Price')
        plt.plot(self.predictions, color='blue',label='Predicted Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show() 