import unittest

from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.distribution import GeometricDistribution
from prolothar_queue_mining.model.service_time import ServiceTimeWithDistribution
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.arrival_process import RecordingArrival
from prolothar_queue_mining.model.arrival_process import ArrivalWithDistribution
from prolothar_queue_mining.model.population import InfinitePopulation
from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.exit import ListCollectorExit
from prolothar_queue_mining.model.waiting_area import FastFirstComeFirstServeWaitingArea
from prolothar_queue_mining.model.waiting_area import FastLastComeFirstServeWaitingArea
from prolothar_queue_mining.inference.queue.waiting_area import NaiveLifoOrFifoWaitingAreaEstimator

class TestNaiveLifoOrFifoEstimator(unittest.TestCase):

    def test_infer_fifo(self):
        ground_truth = Queue(
            RecordingArrival(ArrivalWithDistribution(
                InfinitePopulation(), GeometricDistribution(1/5, seed=1)
            )),
            [Server(ServiceTimeWithDistribution(GeometricDistribution(1/5, seed=2)))],
            waiting_area=FastFirstComeFirstServeWaitingArea(),
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

        discovered_waiting_area = NaiveLifoOrFifoWaitingAreaEstimator().infer_waiting_area(observed_arrivals, observed_departures)
        self.assertEqual('FCFS', discovered_waiting_area.get_discipline_name())

    def test_infer_lifo(self):
        ground_truth = Queue(
            RecordingArrival(ArrivalWithDistribution(
                InfinitePopulation(), GeometricDistribution(1/5, seed=1)
            )),
            [Server(ServiceTimeWithDistribution(GeometricDistribution(1/5, seed=2)))],
            waiting_area=FastLastComeFirstServeWaitingArea(),
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

        discovered_waiting_area = NaiveLifoOrFifoWaitingAreaEstimator().infer_waiting_area(observed_arrivals, observed_departures)
        self.assertEqual('LCFS', discovered_waiting_area.get_discipline_name())

if __name__ == '__main__':
    unittest.main()
