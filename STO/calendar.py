from STO.stock_df import df
import requests

class EconomicCalendar(df):
    def __init__(self):
        super().__init__()
        self.api_key = self.economiccalendarkey

    def get_upcoming_events(self):
        api_url = f'https://api.tradingeconomics.com/calendar?c={self.api_key}'
        response = requests.get(api_url)
        events = response.json()
        return events