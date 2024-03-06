import unittest

from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.population import ListPopulation
from prolothar_queue_mining.model.arrival_process import FixedArrival
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.service_time import LoadDependentServiceTime
from prolothar_queue_mining.model.service_time import FixedServiceTime
from prolothar_queue_mining.model.exit import ListCollectorExit
from prolothar_queue_mining.model.job import Job

class TestLoadDependentServiceTime(unittest.TestCase):

    def test_get_service_time(self):
        population = ListPopulation([Job('A'), Job('B'), Job('C'), Job('D'), Job('E'), Job('F')])
        arrival = FixedArrival(population, [1, 2, 5, 5, 5, 20])
        service_time = LoadDependentServiceTime(
            [FixedServiceTime(1), FixedServiceTime(2)], [1])
        servers = [Server(service_time)]
        exit_point = ListCollectorExit()
        queue = Queue(arrival, servers, exit_point=exit_point)

        environment = Environment()
        queue.schedule_next_arrival(environment)
        environment.run_until_event_queue_is_empty()

        expected_departures = [
            (2, Job('A')), (3, Job('B')), (7, Job('C')), (9, Job('D')), (10, Job('E')), (21, Job('F'))
        ]

        self.assertEqual(len(expected_departures), len(exit_point))
        for i in range(len(expected_departures)):
            self.assertEqual(
                expected_departures[i],
                exit_point[i]
            )

if __name__ == '__main__':
    unittest.main()