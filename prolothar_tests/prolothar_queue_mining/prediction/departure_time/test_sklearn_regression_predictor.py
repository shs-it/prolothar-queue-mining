import unittest

from prolothar_queue_mining.prediction.departure_time import SklearnRegressionDepartureTimePredictor
from prolothar_queue_mining.inference.sklearn.job_regression import train_random_forest_cv
from prolothar_queue_mining.model.population import InfinitePopulation

class TestSklearnRegressionDepartureTimePredictor(unittest.TestCase):

    def test_predict(self):
        population = InfinitePopulation(
            categorical_feature_names=['color', 'shape'],
            numerical_feature_names=['size', 'opacity'],
             seed=18022022)
        y_per_job = {
            job: job.features['size'] * 5 for job in
            [population.get_next_job() for _ in range(100)]
        }
        regressor, job_to_vector_transformer = train_random_forest_cv(
            y_per_job, population.get_categorical_feature_names(),
            population.get_numerical_feature_names())
        predictor = SklearnRegressionDepartureTimePredictor(regressor, job_to_vector_transformer)
        predicted_departure_times = predictor.predict([
            (population.get_next_job(), 0) for _ in range(20)
        ])
        self.assertIsInstance(predicted_departure_times, dict)
        self.assertEqual(len(predicted_departure_times), 20)

if __name__ == '__main__':
    unittest.main()
