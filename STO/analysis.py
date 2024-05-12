from stock_df import df
import yfinance as yf
import numpy as np

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