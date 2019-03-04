from utils import _get_between, _get_common_nonnull, get_logger, get_tqdm
from processor import get_labels
import sklearn.metrics as mt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import seaborn as sns
from typing import Dict, Optional, List

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa E402

register_matplotlib_converters()
plt.style.use('seaborn')
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.facecolor'] = 'w'


def get_binary_labels(y):
    x = y.copy()
    x[x > 0] = 1
    x[x <= 0] = 0
    return x


REGRESSION_METRICS = {
    'RMSE': lambda y_true, y_pred:
        np.sqrt(mt.mean_squared_error(y_true, y_pred)),
    'MAE': mt.median_absolute_error,
    'R2': mt.r2_score,
    'Accuracy': lambda y_true, y_pred:
        mt.accuracy_score(get_binary_labels(y_true),
                          get_binary_labels(y_pred)),
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
                rolling_metrics=['squared_error'], labels=None, **label_args):
        logger = get_logger()
        tqdm, ascii = get_tqdm()
        if labels is None:
            labels = get_labels(fromdate=fromdate, todate=todate,
                                is_debug=is_debug, **label_args)
        all_preds = load_predictions(pred_path)

        metrics = []
        rollings = {}
        for rolling_metric in rolling_metrics:
            rollings[rolling_metric] = []

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
                for rolling_metric in rolling_metrics:
                    rollings[rolling_metric].append(
                        AnalyzerSingle.get_rolling_metric(
                            trues, preds, rolling_metric, name=ticker
                        )
                    )

        # Concatenate results as dataframes
        logger.info('Task Finished: %d Attempted and %d Failed.' %
                    (len(loop_tickers), fails))
        metrics = pd.concat(metrics)
        if show_rolling:
            for rolling_metric in rolling_metrics:
                rollings[rolling_metric] = pd.concat(rollings[rolling_metric],
                                                     1)

        if output_file is not None:
            logger.info('Writing to file.. <%s>' % output_file)
            metrics.to_csv(output_file)

            # Currently not handle rolling metrics, as it's still under
            # developement
            #
            # if show_rolling:
            #     rollings.to_csv(output_file)

        return {
            'Metrics': metrics,
            'Rolling': rollings
        }


class Plotter:

    def __init__(self,
                 pred_paths: List[str],
                 item_names: List[str],
                 fromdate: str,
                 todate: str,
                 rolling_metrics: List[str] = ['squared_error'],
                 label_args: Optional[Dict] = None) -> None:

        self.pred_path = pred_paths
        self.item_names = item_names
        self.fromdate = fromdate
        self.todate = todate
        self.rolling_metrics = rolling_metrics

        if label_args is None:
            label_args = {}
        self.label_args = label_args

        self.results: Dict = {}

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
                rolling_metrics=self.rolling_metrics,
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

        res_df = pd.concat(res, 1)
        res_df.columns = ['(%s) %s' % (metric, name)
                          for name in self.item_names]
        return res_df

    def get_distribution_plot(self, metric: str = 'RMSE') -> mpl.figure.Figure:
        rows = int(np.ceil(len(self.item_names) / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(14, rows * 5),
                                 sharex=True, sharey=True)
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

    def get_cumulative_metrics_plot(self,
                                    metric: str = 'squared_error',
                                    q1: float = 0.75,
                                    q2: float = 0.25) -> mpl.figure.Figure:

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        if metric not in REGRESSION_METRICS:
            condition = 'Cumulative'
        else:
            condition = 'Rolling'

        ax.set_title('%s %s Over Time' % (condition, metric), fontweight=700)

        for name, color in zip(self.item_names, self.colors):
            rmean = self.results[name]['Rolling'][metric][self.stocks]\
                 .quantile(.5, 1)
            rup = self.results[name]['Rolling'][metric][self.stocks]\
                .quantile(q1, 1)
            rdown = self.results[name]['Rolling'][metric][self.stocks]\
                .quantile(q2, 1)

            if metric not in REGRESSION_METRICS:
                ax.plot(rmean.cumsum(), color=color, label=name + ' Median')
                ax.plot(rup.cumsum(), color=color, 
                        label=name + ' {:.0%} Percentile'.format(q1), ls='-.')
                ax.plot(rdown.cumsum(), color=color,
                        label=name + ' {:.0%} Percentile'.format(q2), ls=':')
            else:
                ax.plot(rmean, color=color, label=name)

        ax.legend()

        return fig
