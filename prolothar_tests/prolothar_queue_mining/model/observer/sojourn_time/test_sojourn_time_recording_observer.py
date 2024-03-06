import unittest

from prolothar_queue_mining.model.observer.sojourn_time import SojournTimeRecordingObserver
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.population import InfinitePopulation
from prolothar_queue_mining.model.arrival_process import ArrivalWithDistribution
from prolothar_queue_mining.model.distribution import DiscreteDegenerateDistribution
from prolothar_queue_mining.model.distribution import ExponentialDistribution
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.service_time import ServiceTimeWithDistribution
from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.environment import Environment

class TestSojournTimeRecordingObserver(unittest.TestCase):

    def test_add_and_pop_jobs(self):
        observer = SojournTimeRecordingObserver()
        observer.notify(Job('A'), 3, 8)
        observer.notify(Job('B'), 4, 13)
        observer.notify(Job('C'), 7, 16)
        observer.notify(Job('D'), 7, 21)
        observer.notify(Job('E'), 10, 35)

        self.assertEqual(25, observer.get_max_sojourn_time())
        self.assertEqual(12.4, observer.get_mean_sojourn_time())
        self.assertEqual(9, observer.get_median_sojourn_time())
        self.assertEqual(([3,4,7,7,10],[5,9,9,14,25]), observer.get_timeseries_data())

    def test_simulate_fcfs_queue(self):
        observer = SojournTimeRecordingObserver(min_exit_time=8000)
        population = InfinitePopulation(seed=42)
        arrival = ArrivalWithDistribution(population, DiscreteDegenerateDistribution(1))
        servers = [Server(ServiceTimeWithDistribution(ExponentialDistribution(0.2, seed=42)))]
        queue = Queue(arrival, servers, sojourn_time_observer=observer)

        environment = Environment()
        queue.schedule_next_arrival(environment)
        environment.run_timesteps(10000)

        self.assertGreater(len(observer.get_sojourn_times()), 0)
        self.assertLess(6000, observer.get_min_sojourn_time())
        self.assertLess(8000, observer.get_max_sojourn_time())
        self.assertTrue(observer.get_min_sojourn_time() < observer.get_mean_sojourn_time() < observer.get_max_sojourn_time())

if __name__ == '__main__':
    unittest.main()