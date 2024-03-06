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
from prolothar_queue_mining.model.waiting_area import FlifoWaitingArea
from prolothar_queue_mining.model.waiting_area import RandomOrderWaitingArea
from prolothar_queue_mining.model.waiting_area import FastFirstComeFirstServeWaitingArea
from prolothar_queue_mining.inference.queue.waiting_area import FlifoWaitingAreaEstimator

class TestNaiveLifoOrFifoEstimator(unittest.TestCase):

    def test_infer_waiting_area_fifo_on_low_load(self):
        for load_threshold in [2,5,8]:
            ground_truth = Queue(
                RecordingArrival(ArrivalWithDistribution(
                    InfinitePopulation(), GeometricDistribution(1/5, seed=1)
                )),
                [Server(ServiceTimeWithDistribution(GeometricDistribution(1/5, seed=2)))],
                waiting_area=FlifoWaitingArea(load_threshold, fifo_on_low_load=True),
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

            discovered_waiting_area = FlifoWaitingAreaEstimator().infer_waiting_area(observed_arrivals, observed_departures)
            self.assertEqual(f'FLIFO({load_threshold},FIFO->LIFO)', discovered_waiting_area.get_discipline_name())

    def test_infer_waiting_area_fifo_on_high_load(self):
        for load_threshold in [2,5,8]:
            ground_truth = Queue(
                RecordingArrival(ArrivalWithDistribution(
                    InfinitePopulation(), GeometricDistribution(1/5, seed=1)
                )),
                [Server(ServiceTimeWithDistribution(GeometricDistribution(1/5, seed=2)))],
                waiting_area=FlifoWaitingArea(load_threshold, fifo_on_low_load=False),
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

            discovered_waiting_area = FlifoWaitingAreaEstimator().infer_waiting_area(observed_arrivals, observed_departures)
            self.assertEqual(f'FLIFO({load_threshold},LIFO->FIFO)', discovered_waiting_area.get_discipline_name())

    def test_infer_waiting_area_fifo_only(self):
        for nr_of_servers in [1,2,5,8]:
            ground_truth = Queue(
                RecordingArrival(ArrivalWithDistribution(
                    InfinitePopulation(), GeometricDistribution(1/5, seed=1)
                )),
                [
                    Server(ServiceTimeWithDistribution(GeometricDistribution(1/5, seed=i)))
                    for i in range(nr_of_servers)
                ],
                waiting_area=FastFirstComeFirstServeWaitingArea(),
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

            discovered_waiting_area = FlifoWaitingAreaEstimator().infer_waiting_area(observed_arrivals, observed_departures)
            self.assertIsNone(discovered_waiting_area)

    def test_infer_waiting_area_random(self):
        for nr_of_servers in [1,2,5,8]:
            ground_truth = Queue(
                RecordingArrival(ArrivalWithDistribution(
                    InfinitePopulation(), GeometricDistribution(1/5, seed=1)
                )),
                [
                    Server(ServiceTimeWithDistribution(GeometricDistribution(1/5, seed=i)))
                    for i in range(nr_of_servers)
                ],
                waiting_area=RandomOrderWaitingArea(seed=280422),
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

            discovered_waiting_area = FlifoWaitingAreaEstimator().infer_waiting_area(observed_arrivals, observed_departures)
            self.assertIsNone(discovered_waiting_area)

if __name__ == '__main__':
    unittest.main()
