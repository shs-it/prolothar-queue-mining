import unittest

import numpy as np

from prolothar_queue_mining.model.arrival_process import ArrivalWithDistribution
from prolothar_queue_mining.model.distribution import GammaDistribution
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.population import InfinitePopulation

class TestArrivalWithDistribution(unittest.TestCase):

    def test_get_next_job_with_gamma_arrival(self):
        arrival = ArrivalWithDistribution(InfinitePopulation(), GammaDistribution(1, 1/5, seed=42))
        inter_arrival_times = []
        last_arrival_time = 0
        for _ in range(100000):
            current_arrival_time = arrival.get_next_job()[0]
            inter_arrival_times.append(current_arrival_time - last_arrival_time)
            last_arrival_time = current_arrival_time
        actual_mean = np.mean(inter_arrival_times)
        self.assertAlmostEqual(5, actual_mean, delta=0.02)
        self.assertAlmostEqual(1/5, arrival.get_mean_arrival_rate(), delta=0.0001)

if __name__ == '__main__':
    unittest.main()