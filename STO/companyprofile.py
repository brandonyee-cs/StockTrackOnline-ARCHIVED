from STO.stock_df import df
import requests
from bs4 import BeautifulSoup

class CompanyProfile(df):
    def __init__(self, ticker):
        super.__init__(ticker)

    def get_business_model(self):
        url = f'https://finance.yahoo.com/quote/{self.ticker}/profile?p={self.ticker}'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        self.business_model = soup.find('p', class_='Mt(15px) Lh(1.6)').text

    def get_leadership_team(self):
        url = f'https://finance.yahoo.com/quote/{self.ticker}/profile?p={self.ticker}'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        self.team = []
        for row in table.find_all('tr'):
            cols = row.find_all('td')
            (self.team).append({
                'name': cols[0].text,
                'title': cols[1].text,
                'pay': cols[2].text,
                'exercised': cols[3].text,
                'year_born': cols[4].text
            })

    def get_recent_news(self):
        url = f'https://finance.yahoo.com/quote/{self.ticker}?p={self.ticker}&.tsrc=fin-srch'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        self.news = []
        for item in soup.find_all('li', class_='js-stream-content Pos(r)'):
            (self.news).append({
                'headline': item.find('h3').text,
                'summary': item.find('p', class_='Fz(14px) LineClamp(1,1.6em) M(0)').text,
                'time': item.find('span', class_='C($c-fuji-grey-j) Fz(10px) Pos(a) End(0) T(0)').text
            })