import unittest

from itertools import product

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.service_time import ServiceTimeWithDistribution
from prolothar_queue_mining.model.service_time import FixedServiceTime
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.arrival_process import RecordingArrival
from prolothar_queue_mining.model.arrival_process import ArrivalWithDistribution
from prolothar_queue_mining.model.arrival_process import ExponentialDistributedArrival
from prolothar_queue_mining.model.population import InfinitePopulation
from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.exit import ListCollectorExit
from prolothar_queue_mining.model.distribution import GeometricDistribution
from prolothar_queue_mining.model.distribution import PoissonDistribution
from prolothar_queue_mining.model.waiting_area import FastLastComeFirstServeWaitingArea
from prolothar_queue_mining.model.waiting_area import FastFirstComeFirstServeWaitingArea

from prolothar_queue_mining.inference.queue.nr_of_servers import LowerBoundEstimator

class TestLowerBoundEstimator(unittest.TestCase):

    def test_estimate_nr_of_servers_on_lcfs_toy_example(self):
        estimator = LowerBoundEstimator(FastLastComeFirstServeWaitingArea())
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

    def test_estimate_nr_of_servers_on_fcfs_toy_example(self):
        estimator = LowerBoundEstimator(FastFirstComeFirstServeWaitingArea())
        estimated_nr_of_servers = estimator.estimate_nr_of_servers(
            [
                (Job('A'), 10),
                (Job('B'), 42),
                (Job('C'), 55),
                (Job('D'), 57),
                (Job('E'), 98)
            ],
            [
                (Job('A'), 15),
                (Job('B'), 47),
                (Job('D'), 60),
                (Job('C'), 62),
                (Job('E'), 103)
            ],
        )
        self.assertEqual(2, estimated_nr_of_servers)

    def test_estimate_nr_of_servers_on_fcfs_toy_example_with_synchronized_arrival(self):
        estimator = LowerBoundEstimator(FastFirstComeFirstServeWaitingArea())
        estimated_nr_of_servers = estimator.estimate_nr_of_servers(
            [
                (Job('D'), 10),
                (Job('C'), 10),
                (Job('E'), 10),
                (Job('A'), 10),
                (Job('B'), 10),
                (Job('F'), 11),
                (Job('G'), 11),
            ],
            [
                (Job('B'), 15),
                (Job('C'), 47),
                (Job('A'), 60),
                (Job('E'), 72),
                (Job('D'), 103),
                (Job('G'), 107),
                (Job('F'), 109)
            ],
        )
        self.assertEqual(1, estimated_nr_of_servers)

    def test_estimate_nr_of_servers_on_lcfs_toy_example_with_synchronized_arrival(self):
        estimator = LowerBoundEstimator(FastLastComeFirstServeWaitingArea())
        estimated_nr_of_servers = estimator.estimate_nr_of_servers(
            [
                (Job('D'), 10),
                (Job('C'), 10),
                (Job('E'), 10),
                (Job('A'), 10),
                (Job('B'), 10)
            ],
            [
                (Job('B'), 15),
                (Job('C'), 47),
                (Job('A'), 60),
                (Job('E'), 72),
                (Job('D'), 103)
            ],
        )
        self.assertEqual(1, estimated_nr_of_servers)

    def test_estimate_nr_of_servers_on_lcfs_toy_example_2(self):
        estimator = LowerBoundEstimator(FastLastComeFirstServeWaitingArea())
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
        self.assertEqual(1, estimated_nr_of_servers)

    def test_estimate_nr_of_servers_on_lcfs_synthetic_data(self):
        estimator = LowerBoundEstimator(FastLastComeFirstServeWaitingArea())
        for actual_c, random_seed in product([1,2,3], [42, 4711, 13122021]):
            ground_truth = Queue(
                RecordingArrival(ArrivalWithDistribution(
                    InfinitePopulation(), GeometricDistribution(1 / 2, seed=random_seed))),
                [Server(ServiceTimeWithDistribution(GeometricDistribution(1 / (2 * actual_c), seed=random_seed+1)))],
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
            self.assertLessEqual(estimated_c, actual_c)

    def test_estimate_nr_of_servers_on_fcfs_with_batching_synthetic_data(self):
        ground_truth = Queue(
            RecordingArrival(ExponentialDistributedArrival(InfinitePopulation(seed=23112021), 1/5, seed=23112021)),
            [Server(FixedServiceTime(5))],
            exit_point=ListCollectorExit(),
            batch_size_distribution=PoissonDistribution(2, shift=1, seed=23))

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

        estimated_c = LowerBoundEstimator(
            FastFirstComeFirstServeWaitingArea()).estimate_nr_of_servers(
                observed_arrivals, observed_departures)
        self.assertEqual(1, estimated_c)

if __name__ == '__main__':
    unittest.main()