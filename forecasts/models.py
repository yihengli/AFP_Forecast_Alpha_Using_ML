from abc import ABC, abstractmethod
from utils import get_logger, get_tqdm
import numpy as np


class ModelBase(ABC):

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
                                           ascii=ascii, total=len(arr)-1):
            train_x = x[train_end - self.rolling_bars: train_end]
            train_y = y[train_end - self.rolling_bars: train_end]

            self.log("Training from bar %d to bar %d..." %
                     (train_end - self.rolling_bars, train_end - 1))
            model.train(train_x, train_y, **self.train_args)

            predict_x = x[train_end: predict_end]
            predict_y = y[train_end: predict_end]

            self.log("Predicting from bar %d to bar %d..." %
                     (train_end, predict_end - 1))
            predictions.append(
                model.predict(predict_x, predict_y, **self.predict_args))

        return np.concatenate(
            [np.repeat(np.nan, self.rolling_bars), predictions])
