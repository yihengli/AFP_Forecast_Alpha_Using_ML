import os
import pandas as pd
import numpy as np


class YahooProcessor:
    """
    The set of pre-processors to handle data downloaded from Yahoo APIs
    """

    @staticmethod
    def get_default_folder():
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         os.pardir, 'yahoos', 'data'))

    @staticmethod
    def load_data(ticker, folder=None):
        if folder is None:
            folder = YahooProcessor.get_default_folder()

        return pd.read_csv(os.path.join(folder, ticker+'.csv'),
                           index_col='Date', parse_dates=True)

    @staticmethod
    def get_returns(ticker, folder=None, freq='d', fromdate='2000-01-01',
                    todate='2018-12-31', data_col='Adj Close', is_log=False):
        df = YahooProcessor.load_data(ticker, folder)
        df = df.resample(freq).last()
        df = df[(df.index >= fromdate) & (df.index <= todate)]

        if is_log:
            return df[data_col].pct_change()
        else:
            return np.log(1 + df[data_col].pct_change())
