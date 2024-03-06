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

class TestNSection(unittest.TestCase):

    def test_infer_synthetic_fifo_queue_with_two_servers_nsection(self):
        queue_inference = CueMin(search_strategy_name='7-section', verbose=True)

        ground_truth = Queue(
            RecordingArrival(ArrivalWithDistribution(
                InfinitePopulation(seed=2311211),
                GeometricDistribution(0.2, seed=2311212))),
            [
                Server(ServiceTimeWithDistribution(GeometricDistribution(0.1, seed=134221))),
                Server(ServiceTimeWithDistribution(GeometricDistribution(0.1, seed=134222)))
            ],
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
        self.assertEqual(2, discovered_model.get_nr_of_servers())
        self.assertEqual('FCFS', discovered_model.get_waiting_area().get_discipline_name())
        self.assertIn('ServiceTime(GeometricDistribution(', discovered_model.get_service_time_name())

if __name__ == '__main__':
    unittest.main()
