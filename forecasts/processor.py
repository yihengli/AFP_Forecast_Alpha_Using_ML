import os
import pickle
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pandas as pd

from pathos.helpers import ProcessPool as Pool
from pathos.helpers import cpu_count
from utils import get_logger, get_tqdm


def get_labels(task, tickers, folder, freq, fromdate, todate, forward_bars,
               data_col=None, save_cache=False, load_cache=False,
               cache_name=None, is_debug=True, is_multiprocess=False,
               as_pandas=False):
    processor = TaskLabels[task].value
    logger = get_logger()

    if len(tickers) == 0:
        tickers = processor.get_all_tickers()

    if load_cache and cache_name is not None:
        logger.info('Loading Cached Labels From %s...' % cache_name)
        with open(cache_name, 'rb') as handle:
            returns = pickle.load(handle)

    else:
        returns = processor.get_returns(tickers, folder=folder, freq=freq,
                                        fromdate=fromdate, todate=todate,
                                        forward_bars=forward_bars,
                                        data_col=data_col, is_debug=is_debug,
                                        is_multiprocess=is_multiprocess)
        if save_cache and cache_name is not None:
            logger.info('Cache Processed Labels At %s...' % cache_name)
            with open(cache_name, 'wb') as handle:
                pickle.dump(returns, handle)

    def get_pandas_returns(true_values):

        def build_dataframe(v, k):
            df = pd.DataFrame(v)
            df.columns = [k]
            df.reset_index().drop_duplicates(subset='Date').set_index('Date')
            return df

        res = None
        for k, v in true_values.items():
            df = build_dataframe(v, k)
            if res is None:
                res = df
            else:
                res = pd.merge(res, df, left_index=True, right_index=True,
                               how='outer')

        return res

    if as_pandas:
        return get_pandas_returns(returns)

    return returns


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
                    data_col, is_debug, is_multiprocess):
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
        if freq != 'd':
            df = df.resample(freq).last()
        df = df[(df.index >= fromdate) & (df.index <= todate)]

        if forward_bars is None:
            forward_bars = 0

        if forward_bars == 0:
            returns = df[data_col].pct_change()
        else:
            returns = (df[data_col].shift(-forward_bars) / df[data_col] - 1)

        if is_log:
            return np.log(1 + returns)

        return returns

    @staticmethod
    def get_returns(tickers, folder=None, freq='d', fromdate='2000-01-01',
                    todate='2018-12-31', forward_bars=None, data_col=None,
                    is_log=False, is_debug=False, is_multiprocess=False):
        if data_col is None:
            data_col = 'Adj Close'

        res = {}
        tqdm, ascii = get_tqdm()
        logger = get_logger()

        logger.info('Loading Yahoo Labels...')

        def _get_return(ticker, res=res, folder=folder, freq=freq,
                        fromdate=fromdate, todate=todate, is_debug=is_debug,
                        forward_bars=forward_bars, data_col=data_col,
                        is_log=is_log):
            if is_debug:
                logger = get_logger()
                logger.info('Loading %s' % ticker)
            res[ticker] = YahooProcessor._get_return(ticker, folder, freq,
                                                     fromdate, todate,
                                                     forward_bars, data_col,
                                                     is_log)
            return ticker, res[ticker]

        if is_multiprocess:
            logger.info('Initalized Multiprocess To Get Returns...')
            with Pool(cpu_count()) as p:
                res_pool = list(tqdm(p.imap(_get_return, tickers),
                                     total=len(tickers), ascii=ascii))
            res = {item[0]: item[1] for item in res_pool}

        else:
            list(tqdm(map(_get_return, tickers), total=len(tickers),
                      ascii=ascii))

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


class FamaFrenchFiveFactorProcessor(FeatureProcessor):

    @staticmethod
    def get_default_folder():
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir,
                         "F-F_Research_Data_5_Factors_2x3_daily.csv"))

    @staticmethod
    def load_data(ticker=None, folder=None):
        if folder is None:
            folder = FamaFrenchFiveFactorProcessor.get_default_folder()
        df = pd.read_csv(folder)
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        return df.set_index("date")

    @staticmethod
    def get_features(tickers=None, folder=None, freq='d',
                     fromdate='2000-01-01', todate='2018-12-31',
                     data_cols=None):
        data = FamaFrenchFiveFactorProcessor.load_data(tickers, folder)

        if data_cols is not None:
            data = data[data_cols]

        if freq != 'd':
            data = data.resample(freq).apply(
                lambda x: (x + 1).cumprod()[-1]-1)
        data = data[(data.index <= todate) & (data.index >= fromdate)]
        return data


class FamaFrenchThreeFactorProcessor(FeatureProcessor):

    @staticmethod
    def get_default_folder():
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir,
                         "Fama-French Factors - Daily Frequency.csv"))

    @staticmethod
    def load_data(ticker=None, folder=None):
        if folder is None:
            folder = FamaFrenchThreeFactorProcessor.get_default_folder()
        df = pd.read_csv(folder)
        df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
        df = df.set_index("date")
        return df[['mktrf', 'smb', 'hml']]

    @staticmethod
    def get_features(tickers=None, folder=None, freq='d',
                     fromdate='2000-01-01', todate='2018-12-31',
                     data_cols=None):
        data = FamaFrenchThreeFactorProcessor.load_data(tickers, folder)

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
    ff_5 = FamaFrenchFiveFactorProcessor
    ff_3 = FamaFrenchThreeFactorProcessor
