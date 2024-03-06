import numpy as np

from sklearn.linear_model import Lasso

class BarebonesLasso(Lasso):

    def predict_single(self, x) -> float:
        return np.dot(self.coef_, x) + self.intercept_