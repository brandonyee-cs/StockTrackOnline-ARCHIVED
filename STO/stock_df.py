import STO.config as config
import json
import requests
import pandas as pd

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