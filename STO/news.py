from stock_df import df
import requests

class  news(df):
    def __init__(self, ticker) -> None:
        super().__init__(ticker)
        self.interval = 50

    def SA(self):
        api_url = f'https://www.alphavantage.co/query?function=daily&symbol={self.ticker}&market=US&interval={self.interval}&apikey={self.vantagekey}'
        response = requests.get(api_url)
        data = response.json() #Send the GET request and parse the response
        sentiment_data = data['Time Series Sentiment Index'] #Extract the sentiment data
        for date, sentiment in sentiment_data.items(): print(f'Date: {date}, Sentiment Score: {sentiment["sentiment_score"]}')

    def get_news(self):
        api_url = f'https://newsapi.org/v2/everything?q={self.ticker}&apiKey={self.newskey}'
        response = requests.get(api_url)
        articles = response.json()['articles']
        for article in articles:
            print(f'Title: {article["title"]}, URL: {article["url"]}')