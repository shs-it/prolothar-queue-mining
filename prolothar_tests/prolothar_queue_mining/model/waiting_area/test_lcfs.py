import unittest

from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.queue_network import QueueNetwork
from prolothar_queue_mining.model.queue_network.router import StaticRouter
from prolothar_queue_mining.model.population import ListPopulation
from prolothar_queue_mining.model.arrival_process import FixedArrival
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.service_time import FixedServiceTime
from prolothar_queue_mining.model.exit import ListCollectorExit
from prolothar_queue_mining.model.waiting_area import LastComeFirstServeWaitingArea
from prolothar_queue_mining.model.job import Job

class TestLastComeFirstServeWaitingArea(unittest.TestCase):

    def test_add_and_pop_jobs(self):
        waiting_area = LastComeFirstServeWaitingArea()
        waiting_area.add_job(3, Job('A'))
        waiting_area.add_job(7, Job('B'))
        waiting_area.add_job(10, Job('C'))
        waiting_area.add_job(14, Job('D'))

        self.assertEqual(Job('D'), waiting_area.pop_next_job(4))
        self.assertEqual(Job('C'), waiting_area.pop_next_job(3))
        self.assertEqual(Job('B'), waiting_area.pop_next_job(2))
        waiting_area.add_job(17, Job('E'))
        self.assertEqual(Job('E'), waiting_area.pop_next_job(2))
        self.assertEqual(Job('A'), waiting_area.pop_next_job(1))

if __name__ == '__main__':
    unittest.main()