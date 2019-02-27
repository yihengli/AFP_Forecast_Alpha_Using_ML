# flake8: noqa E402

from utils import (_get_between, _get_common_nonnull, append_df_to_excel,
                   get_logger, get_tqdm)
from processor import get_labels
from typing import Iterable, Optional
import sklearn.metrics as mt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib as mpl
mpl.use('Agg')


plt.style.use('seaborn')
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.facecolor'] = 'w'


REGRESSION_METRICS = {
    'RMSE': lambda y_true, y_pred:
        np.sqrt(mt.mean_squared_error(y_true, y_pred)),
    'MAE': mt.median_absolute_error,
    'R2': mt.r2_score,
    'SIZE': lambda y_true, y_pred: len(y_pred)
}


def load_predictions(file_path, is_debug=True):

    logger = get_logger()
    logger.info('Loading Predictions Files From %s' % file_path)

    df = pd.read_csv(file_path)
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
    return df


class AnalyzerSingle:

    @staticmethod
    def get_regression_metrics(y_true, y_pred, remove_na=True, name=0):

        res = {}
        for metric, func in REGRESSION_METRICS.items():
            res[metric] = func(y_true, y_pred)

        return pd.DataFrame(res, index=[name])

    @staticmethod
    def get_rolling_metric(y_true, y_pred, metric='cum_squared_error',
                           name=None):

        if metric == 'cum_abs_error':
            errors = np.abs(y_true - y_pred)
            res = errors.cumsum()
            res.name = name
            return res
        elif metric == 'cum_squared_error':
            errors = (y_true - y_pred)**2
            res = errors.cumsum()
            res.name = name
            return res
        else:
            func = REGRESSION_METRICS[metric]
            res = [np.nan]
            for i in range(1, len(y_true)):
                res.append(func(y_true[:i], y_pred[:i]))

        if name is None:
            name = metric
        return pd.Series(res, index=y_true.index, name=name)


class Analyzer:

    def log(self, text, is_debug=True):
        logger = get_logger()
        if is_debug:
            logger.info(text)

    def analyze(self, pred_path, fromdate, todate, is_debug=False,
                show_rolling=True, output_file=None,
                rolling_metric='cum_squared_error', **label_args):
        logger = get_logger()
        tqdm, ascii = get_tqdm()
        labels = get_labels(fromdate=fromdate, todate=todate,
                            is_debug=is_debug, **label_args)
        all_preds = load_predictions(pred_path)

        metrics = []
        rollings = []

        fails = 0
        loop_tickers = set(labels.keys()).intersection(set(all_preds.columns))
        for ticker in tqdm(loop_tickers, ascii=ascii):
            self.log('Analyzing <%s>' % ticker, is_debug)

            if ticker not in labels.keys():
                logger.warning('<%s> not in label list' % ticker)
                fails += 1
                continue

            # Get true / pred values
            label = labels[ticker]
            trues = _get_between(label, fromdate, todate)

            preds = _get_between(all_preds[ticker], fromdate, todate)
            trues, preds = _get_common_nonnull(trues, preds)

            # Check Data Validations
            if len(preds) == 0:
                logger.error('<%s>: Not Enough Predictions!' % ticker)
                fails += 1
                continue

            # Calculate Regression metrics
            metrics.append(AnalyzerSingle.get_regression_metrics(
                trues, preds, name=ticker
            ))

            # Calculate Rolling Cumulative Errors
            if show_rolling:
                rollings.append(AnalyzerSingle.get_rolling_metric(
                    trues, preds, rolling_metric, name=ticker
                ))

        # Concatenate results as dataframes
        logger.info('Task Finished: %d Attempted and %d Failed.' %
                    (len(loop_tickers), fails))
        metrics = pd.concat(metrics)
        if show_rolling:
            rolling_res = pd.concat(rollings, 1)

        if output_file is not None:
            logger.info('Writing to file.. <%s>' % output_file)
            metrics.to_csv(output_file)

            if show_rolling:
                rollings.to_csv(output_file)

        return {
            'Metrics': metrics,
            'Rolling': rollings
        }
