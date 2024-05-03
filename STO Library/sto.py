import pandas as pd; import time; import requests; import json

class Data():

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

class SI():
    def fastCalc(stock_df, ticker, window, k): 

        stock_df = SI.EMA(stock_df, window)
        stock_df = SI.SMA(stock_df, window)
        stock_df = SI.RSI(stock_df, window)
        stock_df = SI.BOLLINGER_BANDS(stock_df, window, k)
        return stock_df

    def EMA(stock_df, window):

        if not(window): window = 10
        stock_df['EMA'] = stock_df['close'].ewm(span=window).mean()
        return stock_df #Calculate the EMA using the pandas ewm function

    def SMA(stock_df, window): 

        stock_df['SMA'] = stock_df['close'].rolling(window=window).mean()
        return stock_df #Calculate the SMA using the pandas rolling function

    def RSI(stock_df, window): 

        delta = stock_df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0) #Calculate the RSI using the pandas rolling function

        loss_plus_1 = loss + 1
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss; rsi = 100 - (100 / (1 + rs))
        stock_df['RSI'] = rsi
        
        return stock_df #Prevent division by zero
    
    def BOLLINGER_BANDS(stock_df, window, k):

        if not(window) or not(k): window = 20; k = 2
        stock_df['SMA'] = stock_df['close'].rolling(window=window).mean()
        stock_df['RSD'] = stock_df['close'].rolling(window=window).std()
        stock_df['UBB'] = stock_df['SMA'] + k * stock_df['RSD']
        stock_df['LBB'] = stock_df['SMA'] - k * stock_df['RSD']
        
        return stock_df #Calculate the rolling mean (SMA), rolling standard deviation (RSD), Upper Bollinger Band (UBB), and Lower Bollinger Band (LBB)

class AI:
    def SA(ticker):

        api_url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey=demo' #api_url = f'https://www.alphavantage.co/query?function=daily&symbol={ticker}&market=US&interval={interval}&apikey=7ZET74D05LNJ0FOF'
        response = requests.get(api_url)
        data = response.json() #Send the GET request and parse the response
        sentiment_data = data['Time Series Sentiment Index'] #Extract the sentiment data
        for date, sentiment in sentiment_data.items(): print(f'Date: {date}, Sentiment Score: {sentiment["sentiment_score"]}')