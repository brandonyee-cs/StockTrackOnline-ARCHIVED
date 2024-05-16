from STO.stock_df import df
import yfinance as yf
import numpy as np
import pandas as pd
import talib

class stockAnalysis(df):
    def __init__(self, ticker, window = 10) -> None:
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

    def VWAP(self):
        self.stock_df['VWAP'] = (self.stock_df['volume'] * self.stock_df['close']).cumsum() / self.stock_df['volume'].cumsum()

    def RSI(self): 
        delta = self.stock_df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0) #Calculate the RSI using the pandas rolling function
        loss_plus_1 = loss + 1
        avg_gain = gain.rolling(window=self.window).mean()
        avg_loss = loss.rolling(window=self.window).mean()
        rs = avg_gain / avg_loss; rsi = 100 - (100 / (1 + rs))
        self.stock_df['RSI'] = rsi
    
    def ATR(self):
        high_low = self.stock_df['high'] - self.stock_df['low']
        high_close = np.abs(self.stock_df['high'] - self.stock_df['close'].shift())
        low_close = np.abs(self.stock_df['low'] - self.stock_df['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)

        self.stock_df['ATR'] = true_range.rolling(window=self.window).mean()
            
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

    def fibonacci_retracement(self):
        max_price = self.stock_df['high'].max()
        min_price = self.stock_df['low'].min()

        diff = max_price - min_price
        levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        for level in levels:
            self.stock_df['Fibonacci_Level_{}'.format(level)] = max_price - diff * level
            
    def ichimoku_cloud(self):
        high_9 = self.stock_df['high'].rolling(window=9).max()
        low_9 = self.stock_df['low'].rolling(window=9).min()
        self.stock_df['tenkan_sen'] = (high_9 + low_9) / 2

        high_26 = self.stock_df['high'].rolling(window=26).max()
        low_26 = self.stock_df['low'].rolling(window=26).min()
        self.stock_df['kijun_sen'] = (high_26 + low_26) / 2

        high_52 = self.stock_df['high'].rolling(window=52).max()
        low_52 = self.stock_df['low'].rolling(window=52).min()
        self.stock_df['senkou_span_a'] = ((self.stock_df['tenkan_sen'] + self.stock_df['kijun_sen']) / 2).shift(26)
        self.stock_df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)

        self.stock_df['chikou_span'] = self.stock_df['close'].shift(-26)

    def pivot_points(self):
        self.stock_df['pivot_point'] = (self.stock_df['high'] + self.stock_df['low'] + self.stock_df['close']) / 3
        self.stock_df['resistance_1'] = 2 * self.stock_df['pivot_point'] - self.stock_df['low']
        self.stock_df['support_1'] = 2 * self.stock_df['pivot_point'] - self.stock_df['high']
        self.stock_df['resistance_2'] = self.stock_df['pivot_point'] + (self.stock_df['high'] - self.stock_df['low'])
        self.stock_df['support_2'] = self.stock_df['pivot_point'] - (self.stock_df['high'] - self.stock_df['low'])

    def money_flow_index(self, period=14):
        typical_price = (self.stock_df['high'] + self.stock_df['low'] + self.stock_df['close']) / 3
        money_flow = typical_price * self.stock_df['volume']

        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)

        positive_flow_sum = positive_flow.rolling(window=period).sum()
        negative_flow_sum = negative_flow.rolling(window=period).sum()

        money_flow_ratio = positive_flow_sum / negative_flow_sum
        self.stock_df['MFI'] = 100 - (100 / (1 + money_flow_ratio))

    def accumulation_distribution_line(self):
        clv = ((self.stock_df['close'] - self.stock_df['low']) - (self.stock_df['high'] - self.stock_df['close'])) / (self.stock_df['high'] - self.stock_df['low'])
        clv = clv.fillna(0.0)  # replace NaNs with 0
        self.stock_df['ADL'] = (clv * self.stock_df['volume']).cumsum()

    def average_directional_index(self, period=14):
        self.stock_df['ADX'] = talib.ADX(self.stock_df['high'], self.stock_df['low'], self.stock_df['close'], timeperiod=period)

    def commodity_channel_index(self, period=14):
        tp = (self.stock_df['high'] + self.stock_df['low'] + self.stock_df['close']) / 3
        self.stock_df['CCI'] = (tp - tp.rolling(period).mean()) / (0.015 * tp.rolling(period).std())

    def rate_of_change(self, period=14):
        self.stock_df['ROC'] = self.stock_df['close'].pct_change(periods=period)

    def chaikin_oscillator(self, short_period=3, long_period=10):
        adl = self.accumulation_distribution_line()
        self.stock_df['Chaikin'] = talib.EMA(adl, timeperiod=short_period) - talib.EMA(adl, timeperiod=long_period)

    def quickedit(self):
        stockAnalysis.EMA()
        stockAnalysis.SMA()
        stockAnalysis.VWAP()
        stockAnalysis.RSI()
        stockAnalysis.ATR()
        stockAnalysis.BOLLINGER_BANDS()
        stockAnalysis.MACD()
        stockAnalysis.stochastic_oscillator()
        stockAnalysis.OBV()
        stockAnalysis.fibonacci_retracement()
        stockAnalysis.ichimoku_cloud()
        stockAnalysis.pivot_points()
        stockAnalysis.money_flow_index()
        stockAnalysis.accumulation_distribution_line()
        stockAnalysis.average_directional_index()
        stockAnalysis.commodity_channel_index()
        stockAnalysis.rate_of_change()
        stockAnalysis.chaikin_oscillator()
    
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
        
        return info if error != True else None