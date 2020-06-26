import datetime
import pandas as pd
import yfinance as yf
from finta import TA



class GetData:
    """
    Class to retrieve the training data for a given stock
    """
    NUM_DAYS = 30
    INDICATORS = ['EMA', 'RSI', 'MACD']
    NOTUSED_STATE = ['high', 'low', 'open', 'Adj Close', 'volume']

    def __init__(self, stock, train):
        """
        Function to get the past 5 days of data for a stock, minute by minute for training
        For live data, just need todays 1m interval data
        :param stock: symbol of the stock
        """
        if train:
            start =  (datetime.date.today() - datetime.timedelta( self.NUM_DAYS ) )
            end = datetime.datetime.today()
            self.data = yf.download(stock, start=start, end=end, interval='30m')
            self.data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'}, inplace=True)
            print(self.data)
        else:
            start = datetime.date.today()
            end = datetime.datetime.today()  + datetime.timedelta( 1 )
            self.data = yf.download(stock, start=start, end=end, interval='1d')
            self.data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'}, inplace=True)


    def get_indicator_data(self):
        """
        Function that adds the indicators to the data table used in analysis
        Can add whichever indicators you would need
        :return:
        """

        for indicator in self.INDICATORS:
            ind_data = eval('TA.' + indicator + '(self.data)')
            if not isinstance(ind_data, pd.DataFrame):
                ind_data = ind_data.to_frame()
            self.data = self.data.merge(ind_data, left_index=True, right_index=True)

    def update_data(self, symbol):
        start = datetime.date.today()
        end = datetime.datetime.today() + datetime.timedelta(1)
        self.data = yf.download(symbol, start=start, end=end, interval='1m')
        self.data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'},inplace=True)

        return self.format_data()

    def format_data(self):
        """
        Return the data in a form that can be passed into the neural net (numpy array)
        :return:
        """

        # Filter out the other columns and transform into a np array
        state = self.data.drop( self.NOTUSED_STATE, axis=1 )
        self.vec = state.values.flatten()
        return self.vec


