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

class options(rstrading):
    def buyoptionslimit(self, positionEffect, creditOrDebit, price, symbol, quantity, expirationDate, strike):
        rs.orders.order_buy_option_limit(positionEffect, creditOrDebit, price, symbol,quantity,expirationDate, strike, optionType='both', timeInForce='gtc')