import click
from processor import TaskFeatures, TaskLabels
from models import ModelSelections
import forecast as fct
import os


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
@click.option('--label-cache', default=None, show_default=True,
              help='The location of cached file, if provied, labels will be '
                   'directly loaded from the cached files instead of '
                   'processing from scratch. If no cached file can be found, '
                   'the processed labels will cached at the given location')
@click.option('--lags', default=1, show_default=True,
              help='How many lags to shift the label, this might be helpful '
                   'when solving a time-series prediction')
@click.option('--features', default='ff_basic',
              help='Name of features processors',
              type=click.Choice(FeatureList), show_default=True)
@click.option('--config-path', default=None, show_default=True,
              help='Path to a default config file that details hyper params '
                   'settings for machine learning models.')
@click.option('--rolling/--no-rolling', default=False,
              help='Make rolling predictions or standard train-test split',
              show_default=True)
@click.option('--forward-bars', default=0, show_default=True,
              help='When predicting daily returns, if we are going to predict '
                   'n days forward returns instead of one day return')
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
@click.option('--multiprocess/--no-multiprocess', default=False,
              show_default=True, help='Multiprocess when possible')
@click.pass_context
def forecast(ctx, tickers, label_path, feature_path, freq, model, label,
             label_cache, lags, features, config_path, rolling, rolling_bars,
             forward_bars, predict_bars, train_periods, test_periods,
             minimum_train_bars, multiprocess):

    name = ctx.obj['name']
    output = ctx.obj['output']
    debug = ctx.obj['debug']

    fct.forecast(name, output, tickers, label_path, feature_path, freq, label,
                 label_cache, lags, features, model, train_periods,
                 test_periods, config_path, rolling, rolling_bars,
                 forward_bars, predict_bars, minimum_train_bars, debug,
                 is_multiprocess=multiprocess)


if __name__ == '__main__':
    cli(obj={})
