import config
import requests

class EconomicCalendar():
    def __init__(self):
        self.api_key = config.economiccalendar_key

    def get_upcoming_events(self):
        api_url = f'https://api.economiccalendar.com/upcoming?api_key={self.api_key}'
        response = requests.get(api_url)
        events = response.json()
        return events