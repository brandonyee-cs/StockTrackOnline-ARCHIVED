from STO.stock_df import df
import requests
import bs4 as beautifulsoup
import tweepy
from textblob import TextBlob
import praw

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

class SocialMediaSentiment:
    def __init__(self, twitter_keys, reddit_keys):
        twitter_auth = tweepy.OAuthHandler(twitter_keys['consumer_key'], twitter_keys['consumer_secret'])
        twitter_auth.set_access_token(twitter_keys['access_token'], twitter_keys['access_token_secret'])
        self.twitter_api = tweepy.API(twitter_auth)

        self.reddit_api = praw.Reddit(client_id=reddit_keys['client_id'], 
                                      client_secret=reddit_keys['client_secret'], 
                                      user_agent=reddit_keys['user_agent'])

    def get_twitter_sentiment(self, ticker, tweet_count=100):
        tweets = tweepy.Cursor(self.twitter_api.search, q=ticker, lang='en').items(tweet_count)
        sentiment = {'positive': 0, 'neutral': 0, 'negative': 0}

        for tweet in tweets:
            analysis = TextBlob(tweet.text)
            if analysis.sentiment.polarity > 0:
                sentiment['positive'] += 1
            elif analysis.sentiment.polarity == 0:
                sentiment['neutral'] += 1
            else:
                sentiment['negative'] += 1

        return sentiment

    def get_reddit_sentiment(self, ticker, post_count=100):
        posts = self.reddit_api.subreddit('all').search(ticker, limit=post_count)
        sentiment = {'positive': 0, 'neutral': 0, 'negative': 0}

        for post in posts:
            analysis = TextBlob(post.title)
            if analysis.sentiment.polarity > 0:
                sentiment['positive'] += 1
            elif analysis.sentiment.polarity == 0:
                sentiment['neutral'] += 1
            else:
                sentiment['negative'] += 1

        return sentiment