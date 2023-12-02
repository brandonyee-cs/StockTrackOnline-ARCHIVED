from sto import Data
from sto import SI
from sto import AI

#ticker = input("Enter stock ticker: ")
#window = int(input("Enter window: "))
ticker = 'ibm'; window = 15
stock_df = Data.getData(ticker)
stock_df = SI.EMA(stock_df, window)
stock_df = SI.SMA(stock_df, window)
stock_df = SI.RSI(stock_df, window)
print(stock_df)