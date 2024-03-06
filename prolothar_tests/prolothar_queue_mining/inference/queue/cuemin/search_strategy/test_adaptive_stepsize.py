import unittest
import pandas as pd

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.service_time import FixedServiceTime
from prolothar_queue_mining.model.service_time import ServiceTimeWithDistribution
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.arrival_process import RecordingArrival
from prolothar_queue_mining.model.arrival_process import ExponentialDistributedArrival
from prolothar_queue_mining.model.arrival_process import ArrivalWithDistribution
from prolothar_queue_mining.model.population import InfinitePopulation
from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.exit import ListCollectorExit
from prolothar_queue_mining.model.distribution import PoissonDistribution
from prolothar_queue_mining.model.distribution import DiscreteDegenerateDistribution
from prolothar_queue_mining.model.distribution import GeometricDistribution
from prolothar_queue_mining.model.distribution import NegativeBinomialDistribution
from prolothar_queue_mining.model.waiting_area import LastComeFirstServeWaitingArea
from prolothar_queue_mining.model.waiting_area import FastFirstComeFirstServeWaitingArea
from prolothar_queue_mining.model.waiting_area import PriorityClassWaitingArea
from prolothar_queue_mining.inference.queue import CueMin

class TestAdaptiveStepsize(unittest.TestCase):

    def test_infer_queue_toy_example(self):
        queue_inference = CueMin(record_candidates=True, search_strategy_name='adaptive', verbose=True)
        observed_arrivals = [
            (Job('A', features={}), 3),
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
        self.assertEqual(
            [str(c) for c in df.columns],
            ['D', 'B', 'c', 'S', 'L(M)', 'L(D|B)', 'L(D|S)', 'L(D|V_S)', 'L(D|R_S)', 'mdl_score'])
        self.assertGreater(len(df), 0)
        self.assertEqual(len(df), len(queue_inference.get_recorded_candidates()))

    def test_infer_synthetic_fifo_queue_with_one_server(self):
        queue_inference = CueMin(search_strategy_name='adaptive')

        ground_truth = Queue(
            RecordingArrival(ArrivalWithDistribution(
                InfinitePopulation(seed=23112021),
                GeometricDistribution(0.2, seed=23112021))),
            [Server(FixedServiceTime(5))],
            exit_point=ListCollectorExit())

        environment = Environment(verbose=False)
        ground_truth.schedule_next_arrival(environment)
        environment.run_timesteps(1000)

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
        print(discovered_model)
        self.assertEqual(1, discovered_model.get_nr_of_servers())
        self.assertEqual('FCFS', discovered_model.get_waiting_area().get_discipline_name())
        self.assertEqual('ServiceTime(DegenerateDistribution(5))', discovered_model.get_service_time_name())

    def test_infer_synthetic_fifo_queue_with_one_server_negative_binomial_service_time(self):
        queue_inference = CueMin(search_strategy_name='adaptive')

        ground_truth = Queue(
            RecordingArrival(ArrivalWithDistribution(
                InfinitePopulation(), DiscreteDegenerateDistribution(1))),
            [Server(ServiceTimeWithDistribution(NegativeBinomialDistribution(4, 0.2, seed=42)))],
            exit_point=ListCollectorExit())

        environment = Environment(verbose=False)
        ground_truth.schedule_next_arrival(environment)
        environment.run_timesteps(5000)

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
        self.assertIn(
            'ServiceTime(NegativeBinomialDistribution(',
            discovered_model.get_service_time_name())

    def test_infer_synthetic_fifo_queue_with_one_server_geometric_service_time(self):
        queue_inference = CueMin(search_strategy_name='adaptive')

        ground_truth = Queue(
            RecordingArrival(ArrivalWithDistribution(
                InfinitePopulation(), DiscreteDegenerateDistribution(1))),
            [Server(ServiceTimeWithDistribution(GeometricDistribution(0.2, seed=42)))],
            exit_point=ListCollectorExit())

        environment = Environment(verbose=False)
        ground_truth.schedule_next_arrival(environment)
        environment.run_timesteps(1000)

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
        self.assertIn(
            'ServiceTime(GeometricDistribution(',
            discovered_model.get_service_time_name())

    def test_infer_synthetic_fifo_queue_with_one_server_poisson_service_time(self):
        queue_inference = CueMin(search_strategy_name='adaptive')

        ground_truth = Queue(
            RecordingArrival(ArrivalWithDistribution(
                InfinitePopulation(), DiscreteDegenerateDistribution(1))),
            [Server(ServiceTimeWithDistribution(PoissonDistribution(1, shift=1, seed=42)))],
            exit_point=ListCollectorExit())

        environment = Environment(verbose=False)
        ground_truth.schedule_next_arrival(environment)
        environment.run_timesteps(1000)

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
        self.assertIn(
            'ServiceTime(PoissonDistribution(',
            discovered_model.get_service_time_name())

    def test_infer_synthetic_fifo_queue_with_one_server_with_batching(self):
        queue_inference = CueMin(search_strategy_name='adaptive')

        ground_truth = Queue(
            RecordingArrival(ArrivalWithDistribution(
                InfinitePopulation(seed=23112021),
                GeometricDistribution(1/5, seed=23112021))),
            [Server(FixedServiceTime(5))],
            exit_point=ListCollectorExit(),
            batch_size_distribution=PoissonDistribution(2, shift=1, seed=23))

        environment = Environment(verbose=False)
        ground_truth.schedule_next_arrival(environment)
        environment.run_timesteps(1000)

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
        discovered_batch_size_distribution = discovered_model.get_batch_size_distribution()
        print(discovered_batch_size_distribution)
        self.assertIsInstance(discovered_batch_size_distribution, PoissonDistribution)
        self.assertEqual('ServiceTime(DegenerateDistribution(5))', discovered_model.get_service_time_name())
        self.assertAlmostEqual(3, discovered_batch_size_distribution.get_mean(), delta=0.11)

    def test_infer_synthetic_lifo_queue_with_one_server(self):
        queue_inference = CueMin(verbose=True, search_strategy_name='adaptive')

        ground_truth = Queue(
            RecordingArrival(ArrivalWithDistribution(
                InfinitePopulation(), DiscreteDegenerateDistribution(1))),
            [Server(FixedServiceTime(2))],
            exit_point=ListCollectorExit(),
            waiting_area=LastComeFirstServeWaitingArea())

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

        discovered_model = queue_inference.infer_queue(observed_arrivals, observed_departures)
        self.assertEqual(1, discovered_model.get_nr_of_servers())
        self.assertEqual('LCFS', discovered_model.get_waiting_area().get_discipline_name())
        self.assertEqual('ServiceTime(DegenerateDistribution(2))', discovered_model.get_service_time_name())
        discovered_batch_size_distribution = discovered_model.get_batch_size_distribution()
        self.assertIsInstance(discovered_batch_size_distribution, DiscreteDegenerateDistribution)
        self.assertEqual(1, discovered_batch_size_distribution.get_mean())

    def test_infer_synthetic_pqc_queue_with_one_server(self):
        queue_inference = CueMin(
            categorical_attribute_names=['a', 'b', 'c'],
            search_strategy_name='adaptive')

        ground_truth = Queue(
            RecordingArrival(ExponentialDistributedArrival(
                InfinitePopulation(
                    seed=23112021,
                    categorical_feature_names=['a', 'b', 'c'],
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
        environment.run_timesteps(1000)

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
        self.assertEqual('PQ(b,[2,0,1,3],FCFS)', discovered_model.get_waiting_area().get_discipline_name())
        self.assertEqual('ServiceTime(DegenerateDistribution(5))', discovered_model.get_service_time_name())

if __name__ == '__main__':
    unittest.main()
