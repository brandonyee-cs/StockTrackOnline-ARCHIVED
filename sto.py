import time
import config
import json
import requests
import yfinance as yf
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
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey=7ZET74D05LNJ0FOF'
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
    def __init__(self, window, ticker) -> None:
        super().__init__(ticker)
        self.window = window

    def add_conventional_indicators(self):
        info = yf.Ticker(self.ticker).info
        self.stock_df['PE_ratio'] = info['trailingPE']
        self.stock_df['EPS'] = info['trailingEps']
        
    def EMA(self):
        if not(window): window = 10
        self.stock_df['EMA'] = self.stock_df['close'].ewm(span=self.window).mean()

    def SMA(self): 
        self.stock_df['SMA'] = self.stock_df['close'].rolling(window=self.window).mean()

    def RSI(self): 
        delta = self.stock_df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0) #Calculate the RSI using the pandas rolling function
        loss_plus_1 = loss + 1
        avg_gain = gain.rolling(window=self.window).mean()
        avg_loss = loss.rolling(window=self.window).mean()
        rs = avg_gain / avg_loss; rsi = 100 - (100 / (1 + rs))
        self.stock_df['RSI'] = rsi
            
    def BOLLINGER_BANDS(self):
        if not(window) or not(k): window = 20; k = 2
        self.stock_df['SMA'] = self.stock_df['close'].rolling(window=window).mean()
        self.stock_df['RSD'] = self.stock_df['close'].rolling(window=window).std()
        self.stock_df['UBB'] = self.stock_df['SMA'] + k * self.stock_df['RSD']
        self.stock_df['LBB'] = self.stock_df['SMA'] - k * self.stock_df['RSD']
        
    def MACD(self, short_window=12, long_window=26):
        self.stock_df['short_mavg'] = self.stock_df['close'].ewm(span=short_window, adjust=False).mean()
        self.stock_df['long_mavg'] = self.stock_df['close'].ewm(span=long_window, adjust=False).mean()
        self.stock_df['MACD'] = self.stock_df['short_mavg'] - self.stock_df['long_mavg']
        self.stock_df['signal_line'] = self.stock_df['MACD'].ewm(span=9, adjust=False).mean()
    
    def stochastic_oscillator(self, window=14):
        self.stock_df['low_min'] = self.stock_df['low'].rolling(window).min()
        self.stock_df['high_max'] = self.stock_df['high'].rolling(window).max()
        self.stock_df['k'] = 100 * ((self.stock_df['close'] - self.stock_df['low_min']) / (self.stock_df['high_max'] - self.stock_df['low_min']))
    
    def OBV(self):
        self.stock_df['daily_return'] = self.stock_df['close'].diff()
        self.stock_df['direction'] = np.where(self.stock_df['daily_return'] > 0, 1, -1)
        self.stock_df['direction'][0] = 0
        self.stock_df['vol_adj'] = self.stock_df['volume'] * self.stock_df['direction']
        self.stock_df['OBV'] = self.stock_df['vol_adj'].cumsum()

    def quickedit(self):
        stockAnalysis.EMA()
        stockAnalysis.SMA()
        stockAnalysis.RSI()
        stockAnalysis.BOLLINGER_BANDS()
        stockAnalysis.MACD()
        stockAnalysis.stochastic_oscillator()
        stockAnalysis.OBV()
    
    def getstock_df(self):
        return self.stock_df
    
    def stock_screener(self, market_cap_min=None, market_cap_max=None, pe_min=None, pe_max=None):
        data = yf.download(self.ticker, period='1d')
        info = yf.Ticker(self.ticker).info
        market_cap = info['marketCap']
        pe_ratio = info['trailingPE']
        if market_cap_min is not None and market_cap < market_cap_min:
            error = True
        if market_cap_max is not None and market_cap > market_cap_max:
            error = True
        if pe_min is not None and pe_ratio < pe_min:
            error = True
        if pe_max is not None and pe_ratio > pe_max:
            error = True
        
        return info if error != True else 'ERROR'

class publicSentiment(df):
    def __init__(self, ticker) -> None:
        super().__init__(ticker)
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



class LSTM(nn.Module, df):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1, ticker):
        nn.Module.__init__()
        df.__init__(ticker)
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self):
        lstm_out, self.hidden_cell = self.lstm((self.stock_df).view(len(self.stock_df) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(self.stock_df), -1))
        self.predictions = predictions[-1]

    def processdata(self):
        # Load and preprocess the data
        stock_data = self.stock_df; stock_data = stock_data['Close'].values; stock_data = stock_data.reshape(-1, 1)
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTM(train_window, model.hidden_layer_size, 1, 1)
    
    def calculate_loss(self):
        criterion = nn.MSELoss()
        (self.model).eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in (self.dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(self.dataloader)

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