import unittest

from itertools import product

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.service_time import ServiceTimeWithDistribution
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.arrival_process import RecordingArrival
from prolothar_queue_mining.model.arrival_process import ArrivalWithDistribution
from prolothar_queue_mining.model.population import InfinitePopulation
from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.exit import ListCollectorExit
from prolothar_queue_mining.model.distribution import GeometricDistribution
from prolothar_queue_mining.model.waiting_area import FastLastComeFirstServeWaitingArea

from prolothar_queue_mining.inference.queue.nr_of_servers import UpperBoundEstimator

class TestUpperBoundEstimator(unittest.TestCase):

    def test_estimate_nr_of_servers_on_lcfs_toy_example(self):
        estimator = UpperBoundEstimator(FastLastComeFirstServeWaitingArea())
        estimated_nr_of_servers = estimator.estimate_nr_of_servers(
            [
                (Job('A'), 10),
                (Job('B'), 42),
                (Job('C'), 55),
                (Job('D'), 67),
                (Job('E'), 98)
            ],
            [
                (Job('A'), 15),
                (Job('B'), 47),
                (Job('C'), 60),
                (Job('D'), 72),
                (Job('E'), 103)
            ],
        )
        self.assertEqual(1, estimated_nr_of_servers)

    def test_estimate_nr_of_servers_on_lcfs_toy_example_2(self):
        estimator = UpperBoundEstimator(FastLastComeFirstServeWaitingArea())
        estimated_nr_of_servers = estimator.estimate_nr_of_servers(
            [
                (Job('A'), 51),
                (Job('B'), 52),
                (Job('C'), 53),
                (Job('D'), 54)
            ],
            [
                (Job('A'), 55),
                (Job('B'), 55),
                (Job('C'), 55),
                (Job('D'), 55)
            ],
        )
        self.assertEqual(4, estimated_nr_of_servers)

    def test_estimate_nr_of_servers_on_lcfs_synthetic_data(self):
        estimator = UpperBoundEstimator(FastLastComeFirstServeWaitingArea())
        for actual_c, random_seed in product([1,2,3], [42, 4711, 13122021]):
            ground_truth = Queue(
                RecordingArrival(ArrivalWithDistribution(
                    InfinitePopulation(), GeometricDistribution(1 / 2, seed=random_seed))),
                [
                    Server(ServiceTimeWithDistribution(GeometricDistribution(1 / (2 * actual_c), seed=random_seed+1)))
                    for _ in range(actual_c)
                ],
                exit_point=ListCollectorExit(),
                waiting_area=FastLastComeFirstServeWaitingArea())

            environment = Environment(verbose=False)
            ground_truth.schedule_next_arrival(environment)
            environment.run_timesteps(100)

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

            estimated_c = estimator.estimate_nr_of_servers(observed_arrivals, observed_departures)
            self.assertGreaterEqual(estimated_c, actual_c)

if __name__ == '__main__':
    unittest.main()