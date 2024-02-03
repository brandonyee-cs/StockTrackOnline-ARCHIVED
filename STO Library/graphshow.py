import pandas as pd

NVDA_data = pd.read_csv(r"C:\Users\brand\OneDrive\Desktop\NVDA.csv", index_col='Date')
NVDA_data.head()

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime as dt

plt.figure(figsize=(15, 10))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=60)) #Configurable
x_dates = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in NVDA_data.index.values] #update the csv file with unique data in each colum?

plt.plot(x_dates, NVDA_data['High'], label= 'High')
plt.plot(x_dates, NVDA_data['Low'], label= 'Low')
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.legend
plt.gcf().autofmt_xdate()
plt.show()