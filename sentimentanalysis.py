from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions

# ... other code ...

def get_stock_data(ticker):
    # ... other code ...

    if response.status_code == 200:
        # ... other code ...

        # Sentiment Analysis
        sentiment_score = analyze_sentiment(stock_df['close'])
        stock_df['sentiment'] = sentiment_score

        return stock_df

    else:
        print("Failed to get data")
        return None

def analyze_sentiment(stock_prices):
    api_key = 'your_ibm_watson_api_key'
    url = 'https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/your_instance_id'

    authenticator = IAMAuthenticator(api_key)
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2021-08-01',
        authenticator=authenticator
    )

    natural_language_understanding.set_service_url(url)

    sentiment_score = []

    for price in stock_prices:
        try:
            sentiment = natural_language_understanding.analyze(
                text=str(price),
                features=Features(sentiment=SentimentOptions())).get_result()
            sentiment_score.append(sentiment['sentiment']['document']['score'])
        except:
            sentiment_score.append(0)

    return sentiment_score