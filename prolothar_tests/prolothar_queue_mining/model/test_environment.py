import unittest

from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.queue_network import QueueNetwork
from prolothar_queue_mining.model.queue_network.router import StaticRouter
from prolothar_queue_mining.model.population import ListPopulation
from prolothar_queue_mining.model.population import InfinitePopulation
from prolothar_queue_mining.model.arrival_process import FixedArrival
from prolothar_queue_mining.model.arrival_process import ArrivalWithDistribution
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.distribution import DiscreteDegenerateDistribution
from prolothar_queue_mining.model.distribution import ExponentialDistribution
from prolothar_queue_mining.model.distribution import PoissonDistribution
from prolothar_queue_mining.model.service_time import FixedServiceTime
from prolothar_queue_mining.model.service_time import ServiceTimeWithDistribution
from prolothar_queue_mining.model.exit import ListCollectorExit
from prolothar_queue_mining.model.job import Job

class TestEnvironment(unittest.TestCase):

    def test_simulate_simple_queue(self):
        population = ListPopulation([Job('A'), Job('B'), Job('C'), Job('D'), Job('E')])
        arrival = FixedArrival(population, [10, 42, 55, 67, 98])
        servers = [Server(FixedServiceTime(5))]
        exit_point = ListCollectorExit()
        queue = Queue(arrival, servers, exit_point=exit_point)

        environment = Environment()
        queue.schedule_next_arrival(environment)
        environment.run_timesteps(200)
        for i in range(5):
            self.assertEqual(
                (arrival.get_ith_arrival_time(i) + 5, population.get_ith_job(i)),
                exit_point[i])

    def test_simulate_simple_queue_with_poisson_batches(self):
        population = InfinitePopulation()
        arrival = ArrivalWithDistribution(population, DiscreteDegenerateDistribution(10))
        servers = [Server(ServiceTimeWithDistribution(ExponentialDistribution(0.01, seed=52)))]
        exit_point = ListCollectorExit()
        queue = Queue(
            arrival, servers, exit_point=exit_point,
            batch_size_distribution=PoissonDistribution(2, shift=1, seed=23))

        environment = Environment(verbose=False)
        queue.schedule_next_arrival(environment)
        environment.run_timesteps(500)

        job_departure_list = [int(job.job_id) for job in exit_point.get_recording()[0]]
        self.assertEqual(sorted(job_departure_list), job_departure_list)

    def test_simulate_simple_queue_with_fixed_batches(self):
        population = ListPopulation([Job('A'), Job('B'), Job('C'), Job('D'), Job('E')])
        arrival = FixedArrival(population, [10, 42, 55, 67, 98])
        servers = [Server(FixedServiceTime(5))]
        exit_point = ListCollectorExit()
        queue = Queue(arrival, servers, exit_point=exit_point, batch_size_distribution=DiscreteDegenerateDistribution(2))

        environment = Environment(verbose=False)
        queue.schedule_next_arrival(environment)
        environment.run_timesteps(200)
        expected_exit_times = [47, 47, 72, 72]
        self.assertEqual(4, len(exit_point))
        for i in range(4):
            self.assertEqual(
                (expected_exit_times[i], population.get_ith_job(i)),
                exit_point[i])

    def test_run_until(self):
        population = ListPopulation([Job('A'), Job('B'), Job('C'), Job('D'), Job('E')])
        arrival = FixedArrival(population, [10, 42, 55, 67, 98])
        servers = [Server(FixedServiceTime(5))]
        exit_point = ListCollectorExit()
        queue = Queue(arrival, servers, exit_point=exit_point)

        environment = Environment()
        queue.schedule_next_arrival(environment)
        environment.run_until(lambda: len(exit_point) >= 4)
        self.assertEqual(4, len(exit_point))
        for i in range(4):
            self.assertEqual(
                (arrival.get_ith_arrival_time(i) + 5, population.get_ith_job(i)),
                exit_point[i])

    def test_simulate_two_simple_consecutive_queues(self):
        population = ListPopulation([Job('A'), Job('B'), Job('C'), Job('D'), Job('E')])
        arrival = FixedArrival(population, [10, 42, 55, 67, 98])
        exit_point = ListCollectorExit()
        router = StaticRouter({
            job: ['queue1', 'queue2'] for job in population.get_job_list()
        })
        environment = Environment()
        network = QueueNetwork(environment, arrival, exit_point, router)
        network.add_queue_node('queue1', [Server(FixedServiceTime(30)), Server(FixedServiceTime(30))])
        network.add_queue_node('queue2', [Server(FixedServiceTime(5))])

        network.schedule_next_arrival(environment)
        environment.run_timesteps(200)
        expected_exit_times = [45, 77, 90, 107, 133]
        for i in range(5):
            self.assertEqual(
                (expected_exit_times[i], population.get_ith_job(i)),
                exit_point[i])

if __name__ == '__main__':
    unittest.main()
