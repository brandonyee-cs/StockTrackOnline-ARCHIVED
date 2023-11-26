from STOFunctions import SI
from STOFunctions import Data

ticker = input("Enter stock ticker: ")
window = int(input("Enter window: "))

stock_df = Data.getPrice(ticker)
stock_df = SI.EMA(stock_df, window)
stock_df = SI.SMA(stock_df, window)
stock_df = SI.RSI(stock_df, window)
print(stock_df)
