from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import statsmodels.api as sm
from sklearn import ensemble as ens
from sklearn import linear_model as lm
from sklearn.preprocessing import Normalizer
import pandas as pd
import warnings

from skopt import BayesSearchCV
from utils import get_logger, get_tqdm, load_search_cv_config


def get_model(model_name, config_path, is_normalize=False):
    config_args = load_search_cv_config(config_path)

    model = ModelSelections[model_name].value
    model = model["base"](is_normalize=is_normalize, **model["init_args"])
    model.build_model(config_args=config_args)

    return model


class ModelBase(ABC):

    @abstractmethod
    def build_model(self, config_args=None):
        pass

    @abstractmethod
    def train(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def feature_based_metrics(self, columns=None, index=None):
        pass


class RollingMethod:

    def __init__(self, rolling_bars=90, predict_bars=1, task_name='rolling',
                 is_debug=False, train_args=None, predict_args=None):
        self.rolling_bars = rolling_bars
        self.predict_bars = predict_bars
        self.is_debug = is_debug
        self.task_name = task_name

        self.train_args = {} if train_args is None else train_args
        self.predict_args = {} if predict_args is None else predict_args

    def log(self, text):
        if self.is_debug:
            logger = get_logger()
            logger.info("<{}> - {}".format(self.task_name, text))

    def run(self, model, x, y):

        if len(x) <= self.rolling_bars:
            raise Exception("RollingMethod cannot be initialzed due to size "
                            "is smaller than the rolling periods.")

        arr = np.arange(len(x) - self.rolling_bars)
        arr = arr[arr % self.predict_bars == 0] + self.rolling_bars
        arr = np.concatenate([arr, [len(x)]])

        predictions = []
        other_metrics = []
        tqdm, ascii = get_tqdm()
        for train_end, predict_end in zip(arr[:-1], arr[1:]):
            train_x = x[train_end - self.rolling_bars: train_end]
            train_y = y[train_end - self.rolling_bars: train_end]

            self.log("Training from bar %d to bar %d..." %
                     (train_end - self.rolling_bars, train_end - 1))
            model.train(train_x, train_y, **self.train_args)

            predict_x = x[train_end: predict_end]

            self.log("Predicting from bar %d to bar %d..." %
                     (train_end, predict_end - 1))
            predictions.append(
                model.predict(predict_x, **self.predict_args))
            other_metric = model.feature_based_metrics(
                columns=list(range(x.shape[1])), index=[train_end])
            if other_metric is not None:
                other_metrics.append(other_metric)

        predictions = np.concatenate(
            [np.repeat(np.nan, self.rolling_bars),
             np.concatenate(predictions)])
        metrics = pd.concat(other_metrics) if len(other_metrics) > 0 else None
        return predictions, metrics


class SklearnGeneralModel(ModelBase):

    def __init__(self, is_normalize, model, searchCV=False):
        self.is_normalize = is_normalize
        self.model = model
        self.searchCV = searchCV

    def build_model(self, config_args=None):

        if config_args is None:
            config_args = {}

        if not self.searchCV:
            self.model = self.model(**config_args)
        else:
            self.model = BayesSearchCV(estimator=self.model(), **config_args)

    def train(self, x, y):
        if self.is_normalize:
            self.scaler = Normalizer()
            x = self.scaler.fit_transform(x)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            self.model.fit(x, y)

    def predict(self, x):
        if self.is_normalize:
            x = self.scaler.transform(x)
        return self.model.predict(x)

    def feature_based_metrics(self, columns=None, index=None):
        return pd.DataFrame(
            self.model.best_estimator_.feature_importances_,
            index=columns, columns=index).T


class StatsRegressionModel(ModelBase):

    def __init__(self, is_normalize):
        self.is_normalize = is_normalize

    def build_model(self, config_args=None):
        if config_args is None:
            config_args = {}

        if 'fit_intercept' in config_args.keys():
            self.intercept = config_args['fit_intercept']
        else:
            self.intercept = False

    def train(self, x, y):
        if self.is_normalize:
            self.scaler = Normalizer()
            x = self.scaler.fit_transform(x)

        x = sm.add_constant(x, has_constant='add') if self.intercept else x
        self.model = sm.OLS(y, x)
        self.result = self.model.fit()

    def predict(self, x):
        if self.is_normalize:
            x = self.scaler.transform(x)

        x = sm.add_constant(x, has_constant='add') if self.intercept else x
        return np.array(self.result.predict(x))

    def feature_based_metrics(self, columns=None, index=None):
        """
        Will also generate the t-stats for each feature
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return pd.DataFrame(
                self.result.tvalues, index=columns, columns=index).T


class ModelSelections(Enum):

    Regression = {
        'base': StatsRegressionModel,
        'init_args': {}
    }

    ElasticNet = {
        "base": SklearnGeneralModel,
        "init_args": {
            "model": lm.ElasticNet,
            "searchCV": True
        }
    }

    SKGBM = {
        "base": SklearnGeneralModel,
        "init_args": {
            "model": ens.GradientBoostingRegressor,
            "searchCV": True
        }
    }
