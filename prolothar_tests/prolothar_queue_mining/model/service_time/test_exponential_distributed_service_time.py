import unittest

import numpy as np

from prolothar_queue_mining.model.service_time import ExponentialDistributedServiceTime
from prolothar_queue_mining.model.job import Job

class TestExponentialDistributedServiceTime(unittest.TestCase):

    def test_get_service_time(self):
        service_time = ExponentialDistributedServiceTime(1 / 5, seed=42)
        actual_mean = np.mean([
            service_time.get_service_time(Job('A'), 0)
            for _ in range(100000)
        ])
        self.assertAlmostEqual(5, actual_mean, delta=0.02)
        self.assertAlmostEqual(0.07, service_time.compute_probability(5, Job('A'), 0), delta=0.01)
        self.assertAlmostEqual(0.07, service_time.compute_probability(5, Job('B'), 0), delta=0.01)

if __name__ == '__main__':
    unittest.main()