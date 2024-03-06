import unittest

from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.service_time import FixedServiceTime
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.arrival_process import RecordingArrival
from prolothar_queue_mining.model.arrival_process import ExponentialDistributedArrival
from prolothar_queue_mining.model.population import InfinitePopulation
from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.exit import ListCollectorExit
from prolothar_queue_mining.model.waiting_area import FastFirstComeFirstServeWaitingArea
from prolothar_queue_mining.model.waiting_area import PriorityClassWaitingArea
from prolothar_queue_mining.inference.queue.waiting_area import LinearRegressionEstimator

class TestLinearRegressionEstimator(unittest.TestCase):

    def test_infer_with_one_server(self):
        waiting_area_inference = LinearRegressionEstimator(
            ['x', 'y', 'z'], ['a', 'b', 'c'],
            verbose=False, max_nr_of_epochs=100000, seed=332022)

        ground_truth = Queue(
            RecordingArrival(ExponentialDistributedArrival(
                InfinitePopulation(
                    seed=23112021,
                    categorical_feature_names=['a', 'b', 'c'],
                    numerical_feature_names=['x', 'y', 'z'],
                    nr_of_categories=4
                ), 1/5, seed=23112021)
            ),
            [Server(FixedServiceTime(5))],
            waiting_area=PriorityClassWaitingArea(
                'b', [2, 0, 1, 3], FastFirstComeFirstServeWaitingArea
            ),
            exit_point=ListCollectorExit()
        )

        environment = Environment(verbose=False)
        ground_truth.schedule_next_arrival(environment)
        environment.run_timesteps(300)

        observed_arrivals = [
            (job, arrival_time) for job, arrival_time in zip(
                ground_truth.get_arrival_process().get_recorded_jobs(),
                ground_truth.get_arrival_process().get_recorded_arrival_times()
            )
        ]
        observed_departures = [
            (job, exit_time) for job, exit_time in zip(
                *ground_truth.get_exit().get_recording()
            )
        ]

        discovered_waiting_area = waiting_area_inference.infer_waiting_area(observed_arrivals, observed_departures)
        self.assertIn('PR(LinearRegression', discovered_waiting_area.get_discipline_name())

if __name__ == '__main__':
    unittest.main()
