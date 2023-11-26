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

stock_df = Data.getPrice('nvda')
stock_df = SI.EMA(stock_df, 15)
stock_df = SI.SMA(stock_df, 15)
stock_df = SI.RSI(stock_df, 15)
print(stock_df)
