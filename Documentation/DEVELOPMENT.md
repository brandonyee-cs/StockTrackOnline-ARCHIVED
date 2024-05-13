## Development:
[Back to Read Me](https://github.com/brandonyee-cs/StockTrackOnline)
### APIs: 

  - **Alpha Vantage API**: Used in the [`news`](STO/news.py) class in [STO/news.py](STO/news.py) to get the sentiment score for a given stock symbol. This score is based on the analysis of recent news articles and social media posts about the stock.

  - **News API**: Also used in the [`news`](STO/news.py) class in [STO/news.py](STO/news.py) to retrieve recent news articles about a given stock symbol.

### Web scraping (Beautiful Soup):  

- **Company Profile**: The [`CompanyProfile`](STO/companyprofile.py) class in [STO/companyprofile.py](STO/companyprofile.py) uses web scraping to gather detailed information about a given company. This includes the company's name, industry, description, and more.

- **News**: The [`news`](STO/news.py) class in [STO/news.py](STO/news.py) uses web scraping to gather recent news articles about a given stock symbol. This information is then used to analyze the public sentiment towards the stock.

### Libraries:

Requests, Pandas, Numpy, Pytorch, yFinance, Matplotlib, Flask, Sklearn, Beautiful Soup, PRAW, Tweepy, and TextBlob.
