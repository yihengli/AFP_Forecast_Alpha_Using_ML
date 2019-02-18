from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from sklearn import ensemble as ens
from sklearn import linear_model as lm

from skopt import BayesSearchCV
from utils import get_logger, get_tqdm, load_search_cv_config


def get_model(model_name, config_path):
    config_args = load_search_cv_config(config_path)

    model = ModelSelections[model_name].value
    model = model["base"](**model["init_args"])
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
    def predict(self, x, y):
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
        tqdm, ascii = get_tqdm()
        for train_end, predict_end in tqdm(zip(arr[:-1], arr[1:]),
                                           ascii=ascii, total=len(arr)-1,
                                           disable=(not self.is_debug)):
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

        return np.concatenate(
            [np.repeat(np.nan, self.rolling_bars),
             np.concatenate(predictions)])


class SklearnGeneralModel(ModelBase):

    def __init__(self, model, searchCV=False):
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
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)


class ModelSelections(Enum):

    Regression = {
        "base": SklearnGeneralModel,
        "init_args": {
            "model": lm.LinearRegression,
            "searchCV": False
        }
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
