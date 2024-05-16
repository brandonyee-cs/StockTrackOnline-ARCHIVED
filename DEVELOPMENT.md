# Development:
[Back to Read Me](https://github.com/brandonyee-cs/StockTrackOnline)
## APIs: 

  - **Alpha Vantage API**: This API was used in the `df` class in `STO/stock_df.py` to pull live information about stock price. It was also used in the `SA` function in the `news` class in `STO/news.py` to pull the sentiment score for a given stock symbol. This score is based on the analysis of recent news articles and social media posts about the stock.

  - **Trading Economics**: This API was used in the `EconomicCalendar` in `STO/calendar.py` to pull the information regarding upcoming Macro Economic events that may impact trading.

## Web scraping (Beautiful Soup):  

- **Company Profile**: The `CompanyProfile` class in `STO/companyprofile.py` uses web scraping to gather detailed information about a given company from the Yahoo Finance company profile page, relating to the given ticker. This includes the company's name, industry, description, and more.

- **News**: The `news` class in `STO/news.py` uses web scraping to gather recent news articles about a given stock symbol from the Yahoo Finance News Page (relating to the given ticker). This information is then used to analyze the public sentiment towards the stock.

## Libraries:

Requests, Pandas, Numpy, Pytorch, yFinance, Matplotlib, Flask, Sklearn, Beautiful Soup, PRAW, Tweepy, and TextBlob.
