# """"
# This file will be the main point of control for the trader
# This will call upon the models build in trading.py to generate predictions and place buy and sell orders
# """"

import datetime
import json
import pandas as pd
import pytz, holidays
import yfinance as yf
import time
import os
from get_data import GetData


# Global data and DS, dict to keep stocks and their current bullish or bearish status
# True to indicate a bullish status, False to indicate a bearish one
_stock_list = {'AMD':True, 'ATZ.TO':True, 'DOO.TO':True, 'QSR.TO':True}


def is_market_open(now = None):
    tz = pytz.timezone('US/Eastern')
    us_holidays = holidays.US()
    if not now:
        now = datetime.datetime.now(tz)
    openTime = datetime.time(hour = 9, minute = 30, second = 0)
    closeTime = datetime.time(hour = 16, minute = 0, second = 0)
    # If a holiday
    if now.strftime('%Y-%m-%d') in us_holidays:
        return False
    # If before 09:30 or after 16:00
    if (now.time() < openTime) or (now.time() > closeTime):
        return False
    # If it's a weekend
    if now.date().weekday() > 4:
        return False
    return True

class Portfolio:

    """
    The class that represents the users trading portfolio
    Contains information about their balance and functions to process trades

    To retain information about the portfolio when the program stops, it saves all info to a json
    """

    def __init__(self, username):
        """
        Constructor that checks if the username already has an 'account' (json file with their info). If they do
        it pulls the info from there to be used in this class

        Otherwise, we create a new user with a default balance of $10,000 to be traded with

        :param username: the username you would like your json file to be called
        """

        self.filename = username + "_trades.json"
        if os.path.exists(self.filename):
            print("File exists for " + username)

            with open(self.filename, 'r') as f:
                json_obj = json.load(f)

            self.username = username
            self.balance = json_obj['balance']
            self.stocks = json_obj['stocks']

        else:
            print("User doesn't exist, creating a new portfolio")

            self.balance = 10_000 # Initial balance to trade with
            self.username = username
            self.stocks = {}

            self.write_to_json()


    def place_buy_order(self, symbol, quantity, price):
        """
        Function that takes the steps to process a buy
            - remove the amount of: amt = quantity * price, from the users balance
            - add 'quantity' number of shares of 'symbol' (new dictionary key-value to self.stocks)
            - add the total book value (amt) of that specific stock

        :param symbol: symbol of the stock being bought
        :param quantity: number of shares of the stock
        :param price: the price the stock was bought at

        :return: no return value, function will modify the data and rewrite json to store the info
        """

        amt = quantity * price  # The amount of equity you are adding to your current stake in 'symbol'

        if self.balance >= amt:
            self.balance -= amt
            if self.have_stock(symbol):
                 # We have the stock, just add it to our current balance
                self.stocks[symbol]['num_shares'] += quantity
                self.stocks[symbol]['book_value'] += amt
            else:   # We don't currently own the stock, so we need to add it
                self.stocks[symbol] = {'num_shares' : quantity, 'book_value' : amt}
            self.write_to_json()
        else:
            print("Insufficient funds to buy " + str(quantity) + " shares of " + str(symbol) + " at " + str(price))

    def place_sell_order(self, symbol, quantity, price):

        # First make sure we have a sufficient number of shares
        if self.have_stock(symbol):
            if self.stocks[symbol]['num_shares'] >= quantity:
                # When deducting the book value, we take the average price per share,
                # multiply it by the number we are getting rid of, and then subtract from our total
                self.stocks[symbol]['book_value'] -= ( self.stocks[symbol]['book_value'] / self.stocks[symbol]['num_shares'] ) * quantity
                self.stocks[symbol]['num_shares'] -= quantity

                amt = quantity * price  # Get the amount to return to the account
                self.balance += amt
                self.write_to_json()
        else:
            print("We dont have the stock or we tried selling more shares than we own")

    def write_to_json(self):
        f = open(self.filename, "w")

        user_info = {'username': self.username,
                     'balance': self.balance,
                     'stocks': self.stocks}

        json_obj = json.dumps(user_info, indent=4)
        f.write(json_obj)
        f.close()

    def have_stock(self, symbol):
        # We want to check our stocks to see if we have one we are either buying or selling
        if symbol in self.stocks.keys():
            return True
        return False



def filter_for_day_trading(path):
    """
    Filter out any stocks that may not be a good fit for day trading, such as low volume, low price fluctuation, and high price per share

    Parameters
    ----------
    path : `str`
        The path to the csv that contains stock symbol data

    Returns
    -------
    None
        Generates a new csv with the updated symbols, does not return a value
    """
    symbols = pd.read_csv(path)

    # Get the string values for today's date and tomorrow's date
    today = datetime.datetime.today()
    last_week = today - datetime.timedelta(days=7)

    today = today.strftime('%Y-%m-%d')
    last_week = last_week.strftime('%Y-%m-%d')

    symbol_lst = []
    for index, row in symbols.iterrows():
        symbol = row['Symbol']

        data = yf.download(symbol, start=last_week, end=today, interval='1m')
        index = data.index

        if len(index) > 1400: # Need to filter out data where there isn't a lot of information
            # For a week (5 days) that has every minute documented, it accounts for about 1900 entries
            min = data['Close'].min()
            max = data['Close'].max()
            avg = (min + max) / 2
            percent_diff = (max - min) / avg * 100

            vol_mean = data['Volume'].mean()

            if vol_mean > 15000 and percent_diff > 5.0: # Candidate for day trading
                if max < 80:
                    symbol_lst.append(symbol)
                    print(percent_diff, vol_mean, max, symbol)

    df = pd.DataFrame(symbol_lst, columns=['Symbol'])
    df.to_csv('filtered_symbols.csv', index=False, header=True)




if __name__ == "__main__":

    p = Portfolio('lucas')






    # while (is_market_open()):
    #     # Just before this, we need to sleep for 60 seconds to wait for more data
    #     for stock in _stock_list.keys():
    #         t1 =time.time()
    #         bull_bear, price = get_data(stock)
    #         print(stock, bull_bear)
    #
    #         if bull_bear != _stock_list[stock]: # There has been a change in momentum, need to buy or sell
    #             # If we went from True to False => SELL
    #             if _stock_list[stock] == True:
    #                 # DO SELL STUFF
    #                 _stock_list[stock] = False
    #                 print("SELL "+stock+" at : " + str(price) + '-----' + str(datetime.datetime.now()))
    #                 print('\n\n')
    #                 pass
    #             else:
    #                 # If we went from False to True => BUY
    #                 # DO BUY STUFF
    #                 _stock_list[stock] = True
    #                 print("BUY "+stock+" at : " + str(price) + '-----' + str(datetime.datetime.now()))
    #                 print('\n\n')
    #         t2=time.time()
    #         print(t2-t1)
    #     time.sleep(60)


















