import pandas as pd
from valpy.data.equity import get_history
from tqdm import tqdm
import colorlog
import logging
import os


CURRENT_LOC = os.path.join(os.path.abspath(__file__), os.pardir)
FOLDER = os.path.join(CURRENT_LOC, "data")
TICKERS_FILE = os.path.join(CURRENT_LOC, os.pardir, 'ticker_snp500.csv')
LOG_FILE = 'log.txt'


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        colorlog.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def get_logger(name='data'):
    logger = colorlog.getLogger(name)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = TqdmHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(log_color)s %(levelname)-6s%(reset)s %(white)s%(message)s')) # noqa E501
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


if __name__ == '__main__':
    # Create a folder
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)

    # Read Ticker List
    ticker_list = pd.read_csv(os.path.join(CURRENT_LOC, TICKERS_FILE),
                              header=None).iloc[:, 0].tolist()

    # Prepare for loggers
    logger = get_logger()
    renamed_then_work_symbols = []
    failed_symbols = []

    logger.info("{} Symbols will be checked from Yahoo"
                .format(len(ticker_list)))

    # Loop to get data from Yahoo
    for ticker in tqdm(ticker_list, ascii=True):
        try:
            df = get_history(ticker, fromdate='2000-01-01', set_index=True)
            df.to_csv(os.path.join(FOLDER, ticker + '.csv'))
        except Exception:
            logger.warning("<%s>: Original Ticker Not Exist On Yahoo" % ticker)
            try:
                cut_ticker = ticker.split['.'][0]
                df = get_history(cut_ticker, fromdate='2000-01-01',
                                 set_index=True)
                df.to_csv(os.path.join(FOLDER, cut_ticker + '.csv'))
                renamed_then_work_symbols.append(ticker)
            except Exception:
                logger.error("<%s>: Even Split Not Exists On Yahoo" % ticker)
                failed_symbols.append(ticker)

    logger.info("Completed")

    with open(os.path.join(CURRENT_LOC, LOG_FILE), 'w') as f:
        f.write("{} Tickers are found by checking at "
                "the first half:\n\n".format(len(renamed_then_work_symbols)))
        f.write("\n\t".join(renamed_then_work_symbols))

        f.write("\n" + "---" * 10 + "\n\n")

        f.write("{} Tickers are not found on Yahoo\n\t".format(
            len(failed_symbols)))
        f.write("\n\t".join(failed_symbols))
