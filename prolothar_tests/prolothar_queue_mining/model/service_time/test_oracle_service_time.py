import unittest

from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.service_time import OracleServiceTime
from prolothar_queue_mining.model.job import Job

class TestOracleServiceTime(unittest.TestCase):

    def test_get_service_time(self):
        environment = Environment()
        environment.run_timesteps(10)
        service_time = OracleServiceTime(environment, {
            Job('A'): 24,
            Job('B'): 32
        })
        self.assertEqual(14, service_time.get_service_time(Job('A'), 42))
        self.assertEqual(22, service_time.get_service_time(Job('B'), 4711))

if __name__ == '__main__':
    unittest.main()