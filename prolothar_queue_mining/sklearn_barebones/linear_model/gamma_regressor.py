import numpy as np
from math import exp

from sklearn.linear_model import GammaRegressor

class BarebonesGammaRegressor(GammaRegressor):

    def predict_single(self, x) -> float:
        return exp(np.dot(self.coef_, x) + self.intercept_)