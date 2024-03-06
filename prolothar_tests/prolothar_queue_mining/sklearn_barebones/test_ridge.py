import unittest
import time

from sklearn.datasets import load_diabetes

from prolothar_queue_mining.sklearn_barebones.linear_model import BarebonesRidge

class TestBarebonesRidge(unittest.TestCase):

    def test_fit_and_predict(self):
        X, y_train = load_diabetes(return_X_y=True)
        model = BarebonesRidge(random_state=140422)
        model.fit(X, y_train)

        y = model.predict(X)
        for i in range(X.shape[0]):
            self.assertAlmostEqual(model.predict_single(X[i,:]), y[i])

        start_time = time.time()
        for i in range(X.shape[0]):
            model.predict(X[i,:].reshape(1, -1))[0]
        elapsed_time = time.time() - start_time

        start_time = time.time()
        for i in range(X.shape[0]):
            model.predict_single(X[i,:])
        elapsed_time_single = time.time() - start_time

        self.assertLess(elapsed_time_single, elapsed_time)



if __name__ == '__main__':
    unittest.main()