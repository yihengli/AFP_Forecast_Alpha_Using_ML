from abc import ABC, abstractmethod


class ModelBase(ABC):

    @abstractmethod
    def train(self, x, y):
        pass

    @abstractmethod
    def predict(self, x, y):
        pass


class RollingRegressionModel(ModelBase):

    def __init__(self, rolling_period=90, predict_period=1):
        self.rolling_period = rolling_period
        self.predict_period = 1

    def train(self, x, y):
        pass
