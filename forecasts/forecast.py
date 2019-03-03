import os

import numpy as np
import pandas as pd

from models import RollingMethod, get_model
from processor import get_features, get_labels
from utils import _get_between, get_logger, get_tqdm


def train_and_predict(name, label, lags, features, model, train_periods,
                      test_periods, is_rolling=False, rolling_bars=0,
                      forward_bars=0, predict_bars=0, minimum_train_bars=90,
                      is_debug=False):

    # Shift Features based on given lags
    features = features.shift(lags)

    # Force X and Y have the same dates range
    label = label.replace([np.inf, -np.inf], np.nan)
    label.dropna(inplace=True)

    features = features.replace([np.inf, -np.inf], np.nan)
    features.dropna(inplace=True)

    idx = label.index.intersection(features.index)
    label, features = label.loc[idx], features.loc[idx]

    if not is_rolling:
        # Train Test Split
        train_y = _get_between(label, *train_periods)
        test_y = _get_between(label, *test_periods)
        train_x = _get_between(features, *train_periods)
        test_x = _get_between(features, *test_periods)

        # Give up the task where train or test size is not enough
        if train_x.shape[0] < minimum_train_bars:
            raise Exception("Train Size Not Enough")
        if test_x.shape[0] == 0:
            raise Exception("Test Size Not Enough")

        # For non-rolling case, directly train model on entire training set
        model.train(train_x.values, train_y.values)

        preds_is = model.predict(train_x.values)
        preds_oos = model.predict(test_x.values)

        res = pd.Series(np.concatenate((preds_is, preds_oos)),
                        index=train_y.index.union(test_y.index))

        other_metrics = model.feature_based_metrics(features.columns, [name])

    # For rolling case, train-test split won't be executed
    else:
        label = _get_between(label, train_periods[0], test_periods[1])
        features = _get_between(features, train_periods[0], test_periods[1])

        rolling = RollingMethod(rolling_bars=rolling_bars,
                                predict_bars=predict_bars,
                                task_name=name, is_debug=is_debug)

        preds, other_metrics = rolling.run(model, features.values,
                                           label.values)

        if other_metrics is not None:
            other_metrics.index = name + ' ' + \
                label.index[other_metrics.index].strftime('%Y-%m-%d')

        res = pd.Series(preds, index=label.index)

    return res, other_metrics


def forecast(name, output, tickers, label_path, feature_path, freq, label,
             label_cache, lags, features, model, train_periods, test_periods,
             config_path, is_rolling=False, rolling_bars=0, forward_bars=0,
             predict_bars=0, minimum_train_bars=90, is_debug=False,
             is_multiprocess=False, is_normalize=False):
    logger = get_logger()
    tqdm, ascii = get_tqdm()

    logger.info('<%s> Initialized' % name)

    # Decide cache information
    if label_cache is None:
        save_cache, load_cache, cache_name = False, False, None
    elif os.path.isfile(label_cache):
        save_cache, load_cache, cache_name = False, True, label_cache
    else:
        save_cache, load_cache, cache_name = True, False, label_cache

    labels = get_labels(label, tickers, label_path, freq,
                        train_periods[0], test_periods[1], forward_bars,
                        save_cache=save_cache, load_cache=load_cache,
                        cache_name=cache_name, is_debug=is_debug,
                        is_multiprocess=is_multiprocess)
    features = get_features(features, tickers, feature_path,
                            freq, train_periods[0], test_periods[1])

    if isinstance(features, pd.DataFrame):
        features = {ticker: features for ticker in labels.keys()}
    else:
        commons = set(features.keys()).intersection(set(labels.keys()))
        labels = {k: labels[k] for k in labels.keys() if k in commons}
        features = {k: features[k] for k in features.keys() if k in commons}

    logger.info('<%s> Building Models On %d Stocks...' %
                (name, len(labels.keys())))

    res = {}
    metrics = pd.DataFrame()
    fails = 0

    for k in tqdm(list(labels.keys()), ascii=ascii):

        if is_debug:
            logger.info('<%s> Running On [%s]' % (name, k))

        _label = labels[k]
        _feature = features[k]
        _model = get_model(model, is_normalize=is_normalize,
                           config_path=config_path)

        try:
            pred, other_metrics = train_and_predict(
                k, _label, lags, _feature, _model, train_periods,
                test_periods, is_rolling, rolling_bars, forward_bars,
                predict_bars, minimum_train_bars, is_debug)
            res[k] = pred
            if other_metrics is not None:
                metrics = metrics.append(other_metrics)
        except Exception as e:
            logger.error('<%s> Failed On [%s] Due To "%s"' % (name, k, e))
            fails += 1

    logger.info('<%s> Forecasting Finished, Successfully Built Model on %d '
                'stocks, Failed On %d Stocks.'
                % (name, len(labels.keys()) - fails, fails))

    output_file = os.path.join(output, name + '.csv')
    logger.info('Writing Results To Output Files: %s' % output_file)
    pd.concat(res, 1).to_csv(output_file, index_label='Date')

    if len(metrics) > 0:
        output_metrics = os.path.join(output, name + '_metrics' + '.csv')
        logger.info('Writing Feature Based Metrics To Output Files %s' %
                    output_metrics)
        metrics.columns = next(iter(features.values())).columns
        metrics.to_csv(output_metrics, index_label='Date')

    logger.info('<%s> ******* Done *******' % name)

    return {
        "predictions": res,
        "metrics": metrics
    }
