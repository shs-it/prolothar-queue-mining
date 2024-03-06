import unittest

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.queue import Queue

from prolothar_queue_mining.prediction.departure_time import QueueDepartureTimePredictor
from prolothar_queue_mining.model.distribution import DiscreteDegenerateDistribution
from prolothar_queue_mining.model.distribution import PoissonDistribution
from prolothar_queue_mining.model.service_time import FixedServiceTime
from prolothar_queue_mining.model.service_time import ServiceTimeWithDistribution
from prolothar_queue_mining.model.arrival_process import NullArrival
from prolothar_queue_mining.model.server import Server

class TestQueueDepartureTimePredictor(unittest.TestCase):

    def test_predict(self):
        expected_departure_times = {
            Job('A'): 47,
            Job('B'): 47,
            Job('C'): 72,
            Job('D'): 72,
            Job('E'): 114
        }
        predictor = QueueDepartureTimePredictor(Queue(
            NullArrival(), [Server(FixedServiceTime(5))],
            batch_size_distribution=DiscreteDegenerateDistribution(2)))
        predicted_departure_times = predictor.predict([
            (Job('A'), 10),
            (Job('B'), 42),
            (Job('C'), 55),
            (Job('D'), 67),
            (Job('E'), 98)
        ])
        self.assertDictEqual(expected_departure_times, predicted_departure_times)

    def test_predict_waiting_and_departure_times(self):
        expected_departure_times = {
            Job('A'): 47,
            Job('B'): 47,
            Job('C'): 72,
            Job('D'): 72,
            Job('E'): 114
        }
        expected_waiting_times = {
            Job('A'): 32,
            Job('B'): 0,
            Job('C'): 12,
            Job('D'): 0,
            Job('E'): 11
        }
        predictor = QueueDepartureTimePredictor(Queue(
            NullArrival(), [Server(FixedServiceTime(5))],
            batch_size_distribution=DiscreteDegenerateDistribution(2)))
        predicted_waiting_times, predicted_departure_times = predictor.predict_waiting_and_departure_times([
            (Job('A'), 10),
            (Job('B'), 42),
            (Job('C'), 55),
            (Job('D'), 67),
            (Job('E'), 98)
        ])
        self.assertDictEqual(expected_waiting_times, predicted_waiting_times)
        self.assertDictEqual(expected_departure_times, predicted_departure_times)

    def test_predict_waiting_and_departure_times_mean_mode(self):
        expected_departure_times = {
            Job('A'): 47,
            Job('B'): 47,
            Job('C'): 72,
            Job('D'): 72,
            Job('E'): 114
        }
        expected_waiting_times = {
            Job('A'): 32,
            Job('B'): 0,
            Job('C'): 12,
            Job('D'): 0,
            Job('E'): 11
        }
        predictor = QueueDepartureTimePredictor(Queue(
            NullArrival(), [Server(ServiceTimeWithDistribution(PoissonDistribution(5)))],
            batch_size_distribution=PoissonDistribution(2)), mode='mean')
        predicted_waiting_times, predicted_departure_times = predictor.predict_waiting_and_departure_times([
            (Job('A'), 10),
            (Job('B'), 42),
            (Job('C'), 55),
            (Job('D'), 67),
            (Job('E'), 98)
        ])
        self.assertDictEqual(expected_waiting_times, predicted_waiting_times)
        self.assertDictEqual(expected_departure_times, predicted_departure_times)

    def test_predict_waiting_and_departure_times_distribution(self):
        expected_departure_times = {
            Job('A'): [5, 2, 1],
            Job('B'): [9, 5, 2],
            Job('C'): [10, 7, 5],
            Job('D'): [14, 7, 6],
            Job('E'): [17, 8, 9]
        }
        expected_waiting_times = {
            Job('A'): [0, 0, 0],
            Job('B'): [3, 0, 0],
            Job('C'): [6, 2, 0],
            Job('D'): [6, 3, 1],
            Job('E'): [9, 2, 1]
        }
        predictor = QueueDepartureTimePredictor(Queue(
            NullArrival(),
            [Server(ServiceTimeWithDistribution(PoissonDistribution(2, seed=210422)))]),
            repetitions=3, seed=32)
        predicted_waiting_times, predicted_departure_times = predictor.predict_waiting_and_departure_times_distribution([
            (Job('A'), 1),
            (Job('B'), 2),
            (Job('C'), 3),
            (Job('D'), 4),
            (Job('E'), 5)
        ])
        self.assertDictEqual(expected_waiting_times, predicted_waiting_times)
        self.assertDictEqual(expected_departure_times, predicted_departure_times)

if __name__ == '__main__':
    unittest.main()
