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

from prolothar_queue_mining.inference.queue.nr_of_servers import COrderLcfs

class TestCOrderLcfs(unittest.TestCase):

    def test_estimate_nr_of_servers_on_toy_example(self):
        estimated_nr_of_servers = COrderLcfs().estimate_nr_of_servers(
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

if __name__ == '__main__':
    unittest.main()