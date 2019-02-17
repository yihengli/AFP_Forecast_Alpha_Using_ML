import os
import pandas as pd
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from utils import get_tqdm, get_logger


def get_labels(task, tickers, folder, freq, fromdate, todate, forward_bars,
               data_col=None):
    processor = TaskLabels[task].value

    if len(tickers) == 0:
        tickers = processor.get_all_tickers()

    return processor.get_returns(tickers, folder=folder, freq=freq,
                                 fromdate=fromdate, todate=todate,
                                 forward_bars=forward_bars,
                                 data_col=data_col)


def get_features(task, tickers, folder, freq, fromdate, todate,
                 data_cols=None):
    processor = TaskFeatures[task].value

    return processor.get_features(tickers, folder=folder, freq=freq,
                                  fromdate=fromdate, todate=todate,
                                  data_cols=data_cols)


class LabelProcessor(ABC):

    @staticmethod
    @abstractmethod
    def get_returns(tickers, folder, freq, fromdate, todate, forward_bars,
                    data_col):
        pass

    @staticmethod
    @abstractmethod
    def get_all_tickers(folder):
        pass


class FeatureProcessor(ABC):

    @staticmethod
    @abstractmethod
    def get_features(tickers, folder, fromdate, todate, data_cols):
        raise NotImplementedError


class YahooProcessor(LabelProcessor):
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
    def _get_return(ticker, folder=None, freq='d', fromdate='2000-01-01',
                    todate='2018-12-31', forward_bars=None,
                    data_col='Adj Close', is_log=False):

        df = YahooProcessor.load_data(ticker, folder)
        df = df.resample(freq).last()
        df = df[(df.index >= fromdate) & (df.index <= todate)]

        if not is_log:
            returns = df[data_col].pct_change()
        else:
            returns = np.log(1 + df[data_col].pct_change())

        if forward_bars is not None and forward_bars > 0:
            if is_log:
                returns = returns.rolling(forward_bars)\
                    .apply(lambda x: x.cumsum()[-1] - 1, raw=False)
            else:
                returns = returns.rolling(forward_bars)\
                    .apply(lambda x: (1 + x).cumprod()[-1] - 1, raw=False)

        return returns

    @staticmethod
    def get_returns(tickers, folder=None, freq='d', fromdate='2000-01-01',
                    todate='2018-12-31', forward_bars=None, data_col=None,
                    is_log=False):
        if data_col is None:
            data_col = 'Adj Close'

        res = {}
        tqdm, ascii = get_tqdm()
        logger = get_logger()

        logger.info('Loading Yahoo Labels...')
        for ticker in tqdm(tickers, ascii=ascii):
            logger.info('Loading %s' % ticker)
            res[ticker] = YahooProcessor._get_return(ticker, folder, freq,
                                                     fromdate, todate,
                                                     forward_bars, data_col,
                                                     is_log)
        return res

    @staticmethod
    def get_all_tickers(folder=None):
        if folder is None:
            folder = YahooProcessor.get_default_folder()

        tickers = os.listdir(folder)

        return [t[:-4] for t in tickers if t.endswith('.csv')]


class FamaFrenchBasicProcessor(FeatureProcessor):

    @staticmethod
    def get_default_folder():
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir,
                         "Fama-French Factors - Daily Frequency.csv"))

    @staticmethod
    def load_data(ticker=None, folder=None):
        if folder is None:
            folder = FamaFrenchBasicProcessor.get_default_folder()
        df = pd.read_csv(folder)
        df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
        return df.set_index("date")

    @staticmethod
    def get_features(tickers=None, folder=None, freq='d',
                     fromdate='2000-01-01', todate='2018-12-31',
                     data_cols=None):
        data = FamaFrenchBasicProcessor.load_data(tickers, folder)

        if data_cols is not None:
            data = data[data_cols]

        if freq != 'd':
            data = data.resample(freq).apply(
                lambda x: (x + 1).cumprod()[-1]-1)
        data = data[(data.index <= todate) & (data.index >= fromdate)]
        return data


class TaskLabels(Enum):
    yahoo = YahooProcessor


class TaskFeatures(Enum):
    ff_basic = FamaFrenchBasicProcessor
