import unittest

from prolothar_queue_mining.inference.sklearn.job_regression import train_ridge_regression_cv
from prolothar_queue_mining.inference.sklearn.job_regression import train_gamma_regression_cv
from prolothar_queue_mining.model.population import InfinitePopulation
from prolothar_queue_mining.model.job.regressor import SklearnRegressor
from prolothar_queue_mining.model.distribution import TwoSidedGeometricDistribution

class TestJobRegression(unittest.TestCase):

    def test_train_ridge_regression_cv(self):
        population = InfinitePopulation(
            categorical_feature_names=['color', 'shape'],
            numerical_feature_names=['size', 'opacity'],
             seed=18022022)
        y_per_job = {
            job: job.features['size'] * 5 for job in
            [population.get_next_job() for _ in range(100)]
        }
        regressor, job_to_vector_transformer, distribution = train_ridge_regression_cv(
            y_per_job, population.get_categorical_feature_names(),
            population.get_numerical_feature_names())
        self.assertIsNotNone(regressor)
        self.assertIsNotNone(job_to_vector_transformer)
        self.assertIsNotNone(distribution)

        regressor = SklearnRegressor(regressor, job_to_vector_transformer)
        self.assertIn('Ridge({', str(regressor))

    def test_train_gamma_regression_cv(self):
        population = InfinitePopulation(
            categorical_feature_names=['color', 'shape'],
            numerical_feature_names=['size', 'opacity'],
             seed=18022022)
        y_per_job = {
            job: job.features['size'] * 5 for job in
            [population.get_next_job() for _ in range(100)]
        }
        regressor, job_to_vector_transformer, error_distribution = train_gamma_regression_cv(
            y_per_job, population.get_categorical_feature_names(),
            population.get_numerical_feature_names())
        self.assertIsNotNone(regressor)
        self.assertIsNotNone(job_to_vector_transformer)
        self.assertIsNotNone(error_distribution)
        self.assertIsInstance(error_distribution, TwoSidedGeometricDistribution)

        regressor = SklearnRegressor(regressor, job_to_vector_transformer)
        self.assertIn('GammaRegressor({', str(regressor))

if __name__ == '__main__':
    unittest.main()
