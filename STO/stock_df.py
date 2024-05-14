import csv
import json
import requests
import pandas as pd
import config

class df:
    def __init__(self, ticker) -> None:
        if ticker != None:
            self.stock_df = df.getData(ticker)
            self.ticker = ticker
        self.config = config.config
        self.vantagekey = self.config[0][1]
        self.economiccalendarkey = self.config [1][1]

    def getData(self): #Get stock data for a ticker symbol
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={self.ticker}&outputsize=full&apikey={self.vantagekey}'
        response = requests.get(url) 
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