from sto import Data
from sto import SI
from sto import AI

#ticker = input("Enter stock ticker: ")
#window = int(input("Enter window: "))
ticker = 'nke'; window = 15; k = 2
stock_df = Data.getData(ticker)
stock_df = SI.fastCalc(stock_df, ticker,window,k)
print(stock_df)
