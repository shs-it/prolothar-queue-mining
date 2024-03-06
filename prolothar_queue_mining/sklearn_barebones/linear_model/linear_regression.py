import numpy as np

from sklearn.linear_model import LinearRegression

class BarebonesLinearRegression(LinearRegression):

    def predict_single(self, x) -> float:
        return np.dot(self.coef_, x) + self.intercept_