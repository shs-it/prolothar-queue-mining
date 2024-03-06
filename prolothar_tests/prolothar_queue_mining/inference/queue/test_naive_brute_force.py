import unittest
import pandas as pd

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.service_time.fixed_service_time import FixedServiceTime
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.arrival_process import RecordingArrival
from prolothar_queue_mining.model.arrival_process import ExponentialDistributedArrival
from prolothar_queue_mining.model.population import InfinitePopulation
from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.exit import ListCollectorExit
from prolothar_queue_mining.inference.queue import NaiveBruteForce

class TestNaiveBruteForce(unittest.TestCase):

    def test_infer_queue_toy_example(self):
        queue_inference = NaiveBruteForce([1,2,3], seed=23)
        observed_arrivals = [
            (Job('A'), 3),
            (Job('B'), 4),
            (Job('C'), 5),
            (Job('D'), 6),
            (Job('E'), 7),
            (Job('F'), 8),
        ]
        observed_departues = [
            (Job('A'), 4),
            (Job('B'), 7),
            (Job('C'), 11),
            (Job('D'), 12),
            (Job('E'), 13),
            (Job('F'), 14),
        ]
        inferred_queue = queue_inference.infer_queue(observed_arrivals, observed_departues)
        self.assertIsInstance(inferred_queue, Queue)

        df = queue_inference.get_recording_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual([str(c) for c in df.columns], ['D', 'B', 'c', 'S', 'wasserstein_distance', 'energy_distance', 'MAE'])
        self.assertGreater(len(df), 0)

    def test_infer_synthetic_fifo_queue_with_one_server(self):
        queue_inference = NaiveBruteForce([1,2,3], seed=23)

        ground_truth = Queue(
            RecordingArrival(ExponentialDistributedArrival(InfinitePopulation(seed=23112021), 1/5, seed=23112021)),
            [Server(FixedServiceTime(5))],
            exit_point=ListCollectorExit())

        environment = Environment(verbose=False)
        ground_truth.schedule_next_arrival(environment)
        environment.run_timesteps(50000)

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

        discovered_model = queue_inference.infer_queue(observed_arrivals, observed_departures)
        self.assertEqual(1, discovered_model.get_nr_of_servers())
        self.assertEqual('FCFS', discovered_model.get_waiting_area().get_discipline_name())
        self.assertEqual('ServiceTime(DegenerateDistribution(5))', discovered_model.get_service_time_name())

if __name__ == '__main__':
    unittest.main()