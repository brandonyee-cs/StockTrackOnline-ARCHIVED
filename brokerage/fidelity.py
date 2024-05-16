import fidelity
import requests


class FidelityInformation:
    def __init__(self, ticker, barwidth) -> None:
        url = "https://fidelity-investments.p.rapidapi.com/symbols/get-historical-chart"
        querystring = {"symbols":ticker,"barWidth":barwidth}
        headers = {"X-RapidAPI-Key": "SIGN-UP-FOR-KEY", "X-RapidAPI-Host": "fidelity-investments.p.rapidapi.com"}

''' Depricated
class FidelityBrokerage:
    def __init__(self, username, password):
        self.client = fidelity.Client(username, password)

    def get_account_info(self):
        return self.client.get_account_info()

    def buy_stock(self, ticker, quantity):
        return self.client.place_order(ticker, quantity, fidelity.OrderType.BUY)

    def sell_stock(self, ticker, quantity):
        return self.client.place_order(ticker, quantity, fidelity.OrderType.SELL)

    def get_stock_price(self, ticker):
        return self.client.get_price(ticker)

    def get_portfolio(self):
        return self.client.get_portfolio()
'''
#https://rapidapi.com/apidojo/api/fidelity-investments