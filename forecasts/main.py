import click
from processor import TaskFeatures, TaskLabels
from models import ModelSelections, RollingMethod
from utils import load_search_cv_config, get_logger, get_tqdm
import pandas as pd
import numpy as np
import os


def get_labels(task, tickers, folder, freq, fromdate, todate, data_col=None):
    processor = TaskLabels[task].value

    return processor.get_returns(tickers, folder=folder, freq=freq,
                                 fromdate=fromdate, todate=todate,
                                 data_col=data_col)


def get_features(task, tickers, folder, freq, fromdate, todate,
                 data_cols=None):
    processor = TaskFeatures[task].value

    return processor.get_features(tickers, folder=folder, freq=freq,
                                  fromdate=fromdate, todate=todate,
                                  data_cols=data_cols)


def get_model(model_name, config_path):
    config_args = load_search_cv_config(config_path)

    model = ModelSelections[model_name].value
    model = model["base"](**model["init_args"])
    model.build_model(config_args=config_args)

    return model


def train_and_predict(name, label, features, model, train_periods,
                      test_periods, is_rolling=False, rolling_bars=0,
                      predict_bars=0, minimum_train_bars=90, is_debug=False,
                      label_transforms=None, features_transforms=None):

    if label_transforms is None:
        label_transforms = []
    if features_transforms is None:
        features_transforms = []

    # Transform the data if necessary
    for trans in label_transforms:
        label = trans.apply(label)
    for trans in features_transforms:
        features = trans.apply(features)

    # Force X and Y have the same dates range
    label.dropna(inplace=True)
    features.dropna(inplace=True)

    idx = label.index.intersection(features.index)
    label, features = label.loc[idx], features.loc[idx]

    if not is_rolling:
        # Train Test Split
        train_y = label[(label.index >= train_periods[0]) &
                        (label.index <= train_periods[1])]
        test_y = label[(label.index >= test_periods[0]) &
                       (label.index <= test_periods[1])]
        train_x = features[(features.index >= train_periods[0]) &
                           (features.index <= train_periods[1])]
        test_x = features[(features.index >= test_periods[0]) &
                          (features.index <= test_periods[1])]

        # Give up the task where train or test size is not enough
        if train_x.shape[0] < minimum_train_bars:
            raise Exception("Train Size Not Enough")
        if test_x.shape[0] == 0:
            raise Exception("Test Size Not Enough")

        # For non-rolling case, directly train model on entire training set
        model.train(train_x.values, train_y.values)
        preds_is = model.predict(train_x.values)
        preds_oos = model.predict(test_x.values)

        # import ipdb; ipdb.set_trace()

        res = pd.Series(np.concatenate((preds_is, preds_oos)),
                        index=train_y.index.union(test_y.index))

    # For rolling case, train-test split won't be executed
    else:
        label = label[(label.index >= train_periods[0]) &
                      (label.index <= test_periods[1])]
        features = features[(features.index >= train_periods[0]) &
                            (features.index <= test_periods[1])]

        rolling = RollingMethod(rolling_bars=rolling_bars,
                                predict_bars=predict_bars,
                                task_name=name, is_debug=is_debug)

        preds = rolling.run(model, features.values, label.values)
        res = pd.Series(preds, index=label.index)

    # Reverse engineer the predictions in case it has been transformed before
    for trans in label_transforms[::-1]:
        res = trans.reverse(res)

    return res


def _forecast(name, output, tickers, label_path, feature_path, freq, label,
              features, model, train_periods, test_periods, is_rolling=False,
              rolling_bars=0, predict_bars=0, minimum_train_bars=90,
              is_debug=False, label_transforms=None, features_transforms=None):
    logger = get_logger()
    tqdm, ascii = get_tqdm()

    logger.info('<%s> Initialized' % name)

    labels = get_labels(label, tickers, label_path, freq,
                        train_periods[0], test_periods[1])
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
    for k in tqdm(list(labels.keys()), ascii=ascii):
        logger.info('<%s> Running On [%s]' % (name, k))

        _label = labels[k]
        _feature = features[k]
        _model = get_model('Regression', config_path=None)

        try:
            pred = train_and_predict(
                '', _label, _feature, _model, train_periods,
                test_periods, is_rolling, rolling_bars,
                predict_bars, minimum_train_bars, is_debug,
                label_transforms, features_transforms)
            res[k] = pred
        except Exception as e:
            logger.error('<%s> Failed On [%s] Due To "%s"' % (name, k, e))

    logger.info('<%s> Forecasting Finished, Writing Results To Output File...'
                % name)
    pd.concat(res, 1).to_csv(os.path.join(output, name + '.csv'))

    logger.info('<%s> **** Done ****' % name)


@click.group()
@click.option('--debug/--no-debug', default=False, help="Debug Mode")
@click.option('--name', default='', help='The job name for this runtime')
@click.option('--output', default=os.path.abspath(
    os.path.join(__file__, os.pardir)),
    help='The path to store all the outputs, default to the current '
    'runtime')
@click.pass_context
def cli(ctx, debug, name, output):
    ctx.obj['debug'] = debug
    ctx.obj['name'] = name
    ctx.obj['output'] = output


ModelList = list(map(lambda x: x.name, ModelSelections))
LabelList = list(map(lambda x: x.name, TaskLabels))
FeatureList = list(map(lambda x: x.name, TaskFeatures))


@cli.command()
@click.option('--tickers', '-t', multiple=True, default=None,
              help='Tickers, default to use all the tickers in the path')
@click.option('--label-path', default=None,
              help='The folder path that contains all label data')
@click.option('--feature-path', default=None,
              help='The folder path that contains all feature data')
@click.option('--freq', default='d', help="The frequency to work with, for "
                                          "weekly data, please use `w-Fri`")
@click.option('--model', default='Regression', help='Name of different models',
              type=click.Choice(ModelList), show_default=True)
@click.option('--label', default='yahoo', help='Name of label processors',
              type=click.Choice(LabelList), show_default=True)
@click.option('--features', default='ff_basic',
              help='Name of features processors',
              type=click.Choice(FeatureList), show_default=True)
@click.option('--rolling/--no-rolling', default=False,
              help='Make rolling predictions or standard train-test split',
              show_default=True)
@click.option('--predict-bars', default=30, show_default=True,
              help='How many bars will be used in predict when rolling is on')
@click.option('--rolling-bars', default=90, show_default=True,
              help='How many bars will be used in training when rolling is on')
@click.option('--train-periods', nargs=2, show_default=True,
              help="Training Periods, only effecitve when rolling is off",
              default=('2000-01-01', '2014-12-31'))
@click.option('--test-periods', nargs=2, show_default=True,
              help="Test Periods, only effecitve when rolling is off",
              default=('2015-01-01', '2018-12-31'))
@click.option('--minimum-train-bars', show_default=True, default=90,
              help="Check if train period at least hit this bar, only "
                   "effective when rolling is off")
@click.pass_context
def forecast(ctx, tickers, label_path, feature_path, freq,
             model, label, features, rolling, rolling_bars, predict_bars,
             train_periods, test_periods, minimum_train_bars):

    name = ctx.obj['name']
    output = ctx.obj['output']
    debug = ctx.obj['debug']

    _forecast(name, output, tickers, label_path, feature_path, freq, label,
              features, model, train_periods, test_periods, rolling,
              rolling_bars, predict_bars, minimum_train_bars,
              debug, label_transforms=None, features_transforms=None)


if __name__ == '__main__':
    cli(obj={})
