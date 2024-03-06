from statistics import mean
import unittest
from random import Random

from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.service_time import FixedServiceTime
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.arrival_process import RecordingArrival
from prolothar_queue_mining.model.arrival_process import ArrivalWithDistribution
from prolothar_queue_mining.model.population import InfinitePopulation
from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.exit import ListCollectorExit
from prolothar_queue_mining.model.distribution import DiscreteDegenerateDistribution
from prolothar_queue_mining.model.waiting_area import FastFirstComeFirstServeWaitingArea

from prolothar_queue_mining.inference.queue.batch import LargestGapBatchMiner

class TestPsmBatchMiner(unittest.TestCase):

    def test_group_batches(self):
        random = Random(2522)
        for max_delay in [0, 1, 2]:
            ground_truth = Queue(
                RecordingArrival(ArrivalWithDistribution(
                    InfinitePopulation(), DiscreteDegenerateDistribution(1))),
                [Server(FixedServiceTime(10))],
                exit_point=ListCollectorExit(),
                waiting_area=FastFirstComeFirstServeWaitingArea(),
                batch_size_distribution=DiscreteDegenerateDistribution(3)
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
            i = 0
            observed_departures = []
            for job, exit_time in zip(*ground_truth.get_exit().get_recording()):
                exit_time += (i % 3) * random.randint(0, max_delay)
                observed_departures.append((job, exit_time))
                i += 1
            observed_departures.sort(key=lambda x:x[1])

            batches = LargestGapBatchMiner().group_batches(observed_arrivals, observed_departures)
            self.assertEqual(3, mean(len(batch) for batch in batches), msg=f'max_delay={max_delay}')


if __name__ == '__main__':
    unittest.main()