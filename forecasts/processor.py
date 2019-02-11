import os
import pandas as pd


class YahooProcessor:
    """
    The set of pre-processors to handle data downloaded from Yahoo APIs
    """

    @staticmethod
    def get_default_folder():
        return os.path.join(os.path.dirname(__file__), '..', 'yahoos', 'data')

    @staticmethod
    def load_data(ticker, folder=None):
        if folder is None:
            folder = YahooProcessor.get_default_folder()

        return pd.read_csv(os.path.join(folder, ticker+'.csv'),
                           index_col='Date', parse_dates=True)

    @staticmethod
    def get_returns(ticker, folder=None, freq='d', fromdate='2000-01-01',
                    todate='2018-12-31', data_col='Adj Close'):
        df = YahooProcessor.load_data(ticker, folder)
        pass
