import robin_stocks as rs

class rstrading:
    def __init__(self, username, password):    
        rs.login(username = username, password = password, expiresIn = 86400, by_sms = True) #True if using two factor authentication false is without
    
    def logout(self):
        rs.logout()

    def tradebyprice(self, symbol, ammountInDollars):
        rs.orders.order_buy_fractional_by_price(symbol, ammountInDollars, timeInForce='gtc', extendedHours=False) 
    
    def tradebyshares(self, symbol, shares, pricelimit):
        rs.orders.order_buy_fractional_by_quantity(symbol, shares, pricelimit, timeInForce='gtc', extendedHours=False)

    def sell_stock(self, symbol, shares):
        return rs.orders.order_sell_fractional_by_quantity(symbol, shares, timeInForce='gtc')

    def get_stock_price(self, symbol):
        return rs.stocks.get_latest_price(symbol)[0]

    def get_portfolio_value(self):
        return rs.profiles.load_portfolio_profile()['equity']

    def get_open_stock_positions(self):
        return rs.account.get_open_stock_positions()

class options(rstrading):
    def buyoptionslimit(self, positionEffect, creditOrDebit, price, symbol, quantity, expirationDate, strike):
        rs.orders.order_buy_option_limit(positionEffect, creditOrDebit, price, symbol,quantity,expirationDate, strike, optionType='both', timeInForce='gtc')

    def get_option_chain(self, symbol, expiration_date=None):
        return rs.options.get_chains(symbol, expiration_date)

    def get_option_market_data(self, symbol, expiration_date, strike_price, option_type):
        return rs.options.get_option_market_data(symbol, expiration_date, strike_price, option_type)
    
    def sell_options_limit(self, positionEffect, creditOrDebit, price, symbol, quantity, expirationDate, strike):
        return rs.orders.order_sell_option_limit(positionEffect, creditOrDebit, price, symbol, quantity, expirationDate, strike, optionType='both', timeInForce='gtc')