import numpy as np

from sklearn.linear_model import Ridge

class BarebonesRidge(Ridge):

    def predict_single(self, x) -> float:
        return np.dot(self.coef_, x) + self.intercept_