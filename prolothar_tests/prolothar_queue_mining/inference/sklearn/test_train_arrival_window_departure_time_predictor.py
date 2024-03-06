import unittest

from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.service_time import FixedServiceTime
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.arrival_process import RecordingArrival
from prolothar_queue_mining.model.arrival_process import ArrivalWithDistribution
from prolothar_queue_mining.model.population import InfinitePopulation
from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.exit import ListCollectorExit
from prolothar_queue_mining.model.distribution import GeometricDistribution
from prolothar_queue_mining.inference.sklearn.train_arrival_window_departure_time_predictor import train_random_forest_cv
from prolothar_queue_mining.inference.sklearn.train_arrival_window_departure_time_predictor import train_linear_regression
from prolothar_queue_mining.inference.sklearn.train_arrival_window_departure_time_predictor import train_decision_tree_cv

class TestTrainArrivalWindowDepartureTimePredictor(unittest.TestCase):

    def setUp(self):
        ground_truth = Queue(
            RecordingArrival(ArrivalWithDistribution(
                InfinitePopulation(seed=23112021, categorical_feature_names=['c'], numerical_feature_names=['n']),
                GeometricDistribution(0.2, seed=23112021))),
            [Server(FixedServiceTime(5))],
            exit_point=ListCollectorExit())

        environment = Environment(verbose=False)
        ground_truth.schedule_next_arrival(environment)
        environment.run_timesteps(1000)

        self.observed_arrivals = [
            (job, arrival_time) for job, arrival_time in zip(
                ground_truth.get_arrival_process().get_recorded_jobs(),
                ground_truth.get_arrival_process().get_recorded_arrival_times()
            )
        ]
        self.observed_departures = [
            (job, exit_time) for job, exit_time in zip(
                *ground_truth.get_exit().get_recording()
            )
        ]

    def test_train_linear_regression(self):
        predictor = train_linear_regression(self.observed_arrivals, self.observed_departures, ['c'], ['n'])
        predicted_departures = predictor.predict(self.observed_arrivals)
        self.assertEqual(set(job for job,_ in self.observed_arrivals), set(predicted_departures.keys()))

    def test_train_decision_tree_cv(self):
        predictor = train_decision_tree_cv(self.observed_arrivals, self.observed_departures, ['c'], ['n'])
        predicted_departures = predictor.predict(self.observed_arrivals)
        self.assertEqual(set(job for job,_ in self.observed_arrivals), set(predicted_departures.keys()))

    def test_train_random_forest_cv(self):
        predictor = train_random_forest_cv(self.observed_arrivals, self.observed_departures, ['c'], ['n'])
        predicted_departures = predictor.predict(self.observed_arrivals)
        self.assertEqual(set(job for job,_ in self.observed_arrivals), set(predicted_departures.keys()))
        self.assertEqual(
            [
                'n (-10)', 'c = 0 (-10)', 'c = 1 (-10)', 'c = 2 (-10)', 'relative arrival (-10)',
                'n (-9)', 'c = 0 (-9)', 'c = 1 (-9)', 'c = 2 (-9)', 'relative arrival (-9)',
                'n (-8)', 'c = 0 (-8)', 'c = 1 (-8)', 'c = 2 (-8)', 'relative arrival (-8)',
                'n (-7)', 'c = 0 (-7)', 'c = 1 (-7)', 'c = 2 (-7)', 'relative arrival (-7)',
                'n (-6)', 'c = 0 (-6)', 'c = 1 (-6)', 'c = 2 (-6)', 'relative arrival (-6)',
                'n (-5)', 'c = 0 (-5)', 'c = 1 (-5)', 'c = 2 (-5)', 'relative arrival (-5)',
                'n (-4)', 'c = 0 (-4)', 'c = 1 (-4)', 'c = 2 (-4)', 'relative arrival (-4)',
                'n (-3)', 'c = 0 (-3)', 'c = 1 (-3)', 'c = 2 (-3)', 'relative arrival (-3)',
                'n (-2)', 'c = 0 (-2)', 'c = 1 (-2)', 'c = 2 (-2)', 'relative arrival (-2)',
                'n (-1)', 'c = 0 (-1)', 'c = 1 (-1)', 'c = 2 (-1)', 'relative arrival (-1)',
                'n (0)', 'c = 0 (0)', 'c = 1 (0)', 'c = 2 (0)',
                'n (1)', 'c = 0 (1)', 'c = 1 (1)', 'c = 2 (1)', 'relative arrival (1)',
                'n (2)', 'c = 0 (2)', 'c = 1 (2)', 'c = 2 (2)', 'relative arrival (2)',
                'n (3)', 'c = 0 (3)', 'c = 1 (3)', 'c = 2 (3)', 'relative arrival (3)',
                'n (4)', 'c = 0 (4)', 'c = 1 (4)', 'c = 2 (4)', 'relative arrival (4)',
                'n (5)', 'c = 0 (5)', 'c = 1 (5)', 'c = 2 (5)', 'relative arrival (5)',
                'n (6)', 'c = 0 (6)', 'c = 1 (6)', 'c = 2 (6)', 'relative arrival (6)',
                'n (7)', 'c = 0 (7)', 'c = 1 (7)', 'c = 2 (7)', 'relative arrival (7)',
                 'n (8)', 'c = 0 (8)', 'c = 1 (8)', 'c = 2 (8)', 'relative arrival (8)',
                 'n (9)', 'c = 0 (9)', 'c = 1 (9)', 'c = 2 (9)', 'relative arrival (9)',
                 'n (10)', 'c = 0 (10)', 'c = 1 (10)', 'c = 2 (10)', 'relative arrival (10)'
            ],
            predictor.get_feature_names_for_vector()
        )

if __name__ == '__main__':
    unittest.main()
