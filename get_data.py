import datetime
import pandas as pd
import yfinance as yf
from finta import TA


class GetData:
    """
    Class to retrieve the training data for a given stock
    """
    NUM_DAYS = 5_000
    INDICATORS = ['EMA', 'RSI', 'BBANDS',  'MACD']
    def __init__(self, stock):
        """
        Function to get the past 730 days of data for a stock, hourly
        :param stock:
        """


        start =  (datetime.date.today() - datetime.timedelta( self.NUM_DAYS ) )
        end = datetime.datetime.today()
        self.data = yf.download(stock, start=start, end=end, interval='1d')
        self.data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'}, inplace=True)
        self.data['label'] = 0


    def get_indicator_data(self):

        for indicator in self.INDICATORS:
            ind_data = eval('TA.' + indicator + '(self.data)')
            if not isinstance(ind_data, pd.DataFrame):
                ind_data = ind_data.to_frame()

            self.data = self.data.merge(ind_data, left_index=True, right_index=True)


