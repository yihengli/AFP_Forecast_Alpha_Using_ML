import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class Processor(ABC):

    @abstractmethod
    def get_returns(tickers, folder, freq, fromdate, todate, data_col):
        pass

    @abstractmethod
    def get_all_tickers(folder):
        pass


class YahooProcessor(Processor):
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

    def _get_return(ticker, folder=None, freq='d', fromdate='2000-01-01',
                    todate='2018-12-31', data_col='Adj Close', is_log=False):
        df = YahooProcessor.load_data(ticker, folder)
        df = df.resample(freq).last()
        df = df[(df.index >= fromdate) & (df.index <= todate)]

        if is_log:
            return df[data_col].pct_change()
        else:
            return np.log(1 + df[data_col].pct_change())

    @staticmethod
    def get_returns(tickers, folder=None, freq='d', fromdate='2000-01-01',
                    todate='2018-12-31', data_col='Adj Close', is_log=False):
        res = {}
        for ticker in tickers:
            res[ticker] = YahooProcessor._get_return(ticker, folder, freq,
                                                     fromdate, todate,
                                                     data_col, is_log)
        return res

    @staticmethod
    def get_all_tickers(folder=None):
        if folder is None:
            folder = YahooProcessor.get_default_folder()

        tickers = os.listdir(folder)

        return [t[:-4] for t in tickers if t.endswith('.csv')]
