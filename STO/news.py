from stock_df import df
import requests
import bs4 as beautifulsoup

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
        url = f'https://finance.yahoo.com/quote/{self.ticker}/news?p={self.ticker}'
        response = requests.get(url)
        soup = beautifulsoup(response.text, 'html.parser')
        articles = soup.find_all('article')
        for article in articles:
            title = article.find('h3').text
            url = article.find('a')['href']
            print(f'Title: {title}, URL: {url}')