import pandas as pd
import sklearn.metrics as mt
import numpy as np
from processor import get_labels
from utils import get_logger, get_tqdm, _get_between, _get_common_nonnull,\
    append_df_to_excel


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
    def get_rolling_metric(y_true, y_pred, metric='cum_abs_error', name=None):

        if metric == 'cum_abs_error':
            errors = np.abs(y_true - y_pred)
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

    def analyze(self, pred_path, train_periods, test_periods, is_debug,
                show_rolling=True, show_details=True, output_file=None,
                rolling_metric='cum_abs_error', **label_args):
        logger = get_logger()
        tqdm, ascii = get_tqdm()
        labels = get_labels(fromdate=train_periods[0], todate=test_periods[1],
                            is_debug=is_debug, **label_args)
        preds = load_predictions(pred_path)

        metric_res = {'IS': [], 'OOS': []}
        rolling_res = {'IS': [], 'OOS': []}

        fails = 0
        for ticker in tqdm(preds.columns, ascii=ascii):
            self.log('Analyzing <%s>' % ticker, is_debug)

            if ticker not in labels.keys():
                logger.warning('<%s> not in label list' % ticker)
                fails += 1
                continue

            label = labels[ticker]
            train_true = _get_between(label, *train_periods)
            test_true = _get_between(label, *test_periods)
            train_pred = _get_between(preds[ticker], *train_periods)
            test_pred = _get_between(preds[ticker], *test_periods)

            train_true, train_pred = _get_common_nonnull(train_true,
                                                         train_pred)
            test_true, test_pred = _get_common_nonnull(test_true, test_pred)

            # Check Data Validations
            if len(train_pred) == 0:
                logger.error('<%s>: Not Enough In Sample Predictions!'
                             % ticker)
                fails += 1
                continue
            if len(test_pred) == 0:
                logger.error('<%s>: Not Enough Out Sample Preditions!'
                             % ticker)
                fails += 1
                continue

            # Calculate Regression metrics
            metric_res['IS'].append(AnalyzerSingle.get_regression_metrics(
                train_true, train_pred, name=ticker))
            metric_res['OOS'].append(AnalyzerSingle.get_regression_metrics(
                test_true, test_pred, name=ticker))

            # Calculate Rolling Cumulative Errors
            if show_rolling:
                rolling_res['IS'].append(AnalyzerSingle.get_rolling_metric(
                    train_true, train_pred, rolling_metric, name=ticker))
                rolling_res['OOS'].append(AnalyzerSingle.get_rolling_metric(
                    test_true, test_pred, rolling_metric, name=ticker))

        logger.info('Task Finished: %d Attempted and %d Failed.' %
                    (len(preds.columns), fails))
        metric_res['IS'] = pd.concat(metric_res['IS'])
        metric_res['OOS'] = pd.concat(metric_res['OOS'])

        if show_rolling:
            rolling_res['IS'] = pd.concat(rolling_res['IS'], 1)
            rolling_res['OOS'] = pd.concat(rolling_res['OOS'], 1)

        if output_file is not None:
            logger.info('Writing to file.. <%s>' % output_file)
            metric_res['IS'].to_excel(output_file, sheet_name='Metrics IS')
            append_df_to_excel(output_file, metric_res['OOS'],
                               sheet_name='Metrics OOS')
            if show_rolling:
                append_df_to_excel(output_file, rolling_res['IS'],
                                   sheet_name='Rolling %s IS'
                                   % rolling_metric)
                append_df_to_excel(output_file, rolling_res['OOS'],
                                   sheet_name='Rolling %s OOS'
                                   % rolling_metric)

        return {
            'Metrics_IS': metric_res['IS'],
            'Metrics_OOS': metric_res['OOS'],
            'Rolling_IS': rolling_res['IS'],
            'Rolling_OOS': rolling_res['OOS']
        }
