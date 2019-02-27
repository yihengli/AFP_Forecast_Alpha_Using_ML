from utils import _get_between, _get_common_nonnull, get_logger, get_tqdm
from processor import get_labels
import sklearn.metrics as mt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, Iterable, Optional

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa E402

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
    def get_rolling_metric(y_true, y_pred, metric='squared_error',
                           name=None):

        if metric == 'abs_error':
            errors = np.abs(y_true - y_pred)
            errors.name = name
            return errors
        elif metric == 'squared_error':
            errors = (y_true - y_pred)**2
            errors.name = name
            return errors
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
                rolling_metric='squared_error', labels=None, **label_args):
        logger = get_logger()
        tqdm, ascii = get_tqdm()
        if labels is None:
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
            rollings = pd.concat(rollings, 1)

        if output_file is not None:
            logger.info('Writing to file.. <%s>' % output_file)
            metrics.to_csv(output_file)

            if show_rolling:
                rollings.to_csv(output_file)

        return {
            'Metrics': metrics,
            'Rolling': rollings
        }


class Plotter:

    def __init__(self,
                 pred_paths: Iterable[str],
                 item_names: Iterable[str],
                 fromdate: str,
                 todate: str,
                 rolling_metric: str = 'squared_error',
                 label_args: Optional[Dict] = None) -> None:
        self.pred_path = pred_paths
        self.item_names = item_names
        self.fromdate = fromdate
        self.todate = todate
        self.rolling_metric = rolling_metric

        if label_args is None:
            label_args = {}
        self.label_args = label_args

        self.results = {}

        cmap = sns.color_palette()
        self.colors = [mpl.colors.rgb2hex(c[:3]) for c in cmap]

    @staticmethod
    def _exclude_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
        return df[df[col] <= df[col].quantile(.99) * 5]

    def _set_common_stocks(self) -> None:
        stocks = None
        for name in self.item_names:
            if stocks is None:
                stocks = set(self.results[name]['Metrics'].index)
            else:
                stocks = stocks.intersection(
                    set(self.results[name]['Metrics'].index))

        self.stocks = stocks

    def set_analyzers(self, exclude_outlier: bool = True) -> None:
        analyzer = Analyzer()
        logger = get_logger()

        labels = get_labels(fromdate=self.fromdate, todate=self.todate,
                            is_debug=False, **self.label_args)

        for pred, name in zip(self.pred_path, self.item_names):
            logger.info("Analyzing %s..." % name)

            self.results[name] = analyzer.analyze(
                pred_path=pred,
                fromdate=self.fromdate,
                todate=self.todate,
                rolling_metric=self.rolling_metric,
                labels=labels
            )

            if exclude_outlier:
                self.results[name]["Metrics"] = self._exclude_outliers(
                    self.results[name]["Metrics"], 'RMSE')

        self._set_common_stocks()

    def get_performance_table(self, metric: str = 'RMSE') -> pd.DataFrame:
        res = []
        for name in self.item_names:
            data = self.results[name]['Metrics'][metric]
            data = data.loc[self.stocks]
            res.append(data.describe())

        res = pd.concat(res, 1)
        res.columns = ['(%s) %s' % (metric, name) for name in self.item_names]
        return res

    def get_distribution_plot(self, metric: str = 'RMSE') -> mpl.figure.Figure:
        rows = int(np.ceil(len(self.item_names) / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(14, rows * 5))
        axes = axes.flatten()

        for ax, name, color in zip(axes, self.item_names, self.colors):
            data = self.results[name]['Metrics'][metric]
            data = data.loc[self.stocks]

            data.hist(ax=ax, bins=30, color=color)
            ax.set_title("%s (%s)" % (name, metric), fontweight=700)
            ax.axvline(data.mean(), color='red', ls=':',
                       label="Average %s" % metric)
            ax.annotate('%.5f' % data.mean(), xy=(data.mean(), 0),
                        xytext=(2, 2), textcoords='offset points', color='w')
            ax.legend()

        if len(axes) > len(self.item_names):
            axes[-1].set_visible(False)

        return fig

    def get_cumulative_metrics_plot(self) -> mpl.figure.Figure:

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.set_title('Cumulative %s Over Time' % self.rolling_metric,
                     fontweight=700)

        for name, color in zip(self.item_names, self.colors):
            data = self.results[name]['Rolling'][self.stocks].mean(1)
            if self.rolling_metric not in REGRESSION_METRICS:
                ax.plot(data.cumsum(), color=color, label=name)
            else:
                ax.plot(data, color=color, label=name)

        ax.legend()

        return fig
