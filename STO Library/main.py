from sto import Data
from sto import SI
from sto import AI

#ticker = input("Enter stock ticker: ")
#window = int(input("Enter window: "))
ticker = 'mdb'; window = 15; k = 2
stock_df = Data.getData(ticker)
stock_df = SI.fastCalc(stock_df, ticker,window,k)
print(stock_df)

import io
towrite = io.BytesIO()
stock_df.to_excel(towrite)  # write to BytesIO buffer
towrite.seek(0)  # reset pointer

import requests
# Define the local filename to save data
local_file = r'C:\Users\brand\Downloads\cool.txt'
# Make http request for remote file data
# Save file data to local copy
with open(local_file, 'wb')as file:
    file.write(stock_df)