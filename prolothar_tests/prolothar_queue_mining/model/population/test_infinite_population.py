import unittest

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.population import InfinitePopulation

class TestInfinitePopulation(unittest.TestCase):

    def test_get_next_job_without_features(self):
        population = InfinitePopulation()
        self.assertEqual(Job('0'), population.get_next_job())
        self.assertEqual(Job('1'), population.get_next_job())
        self.assertEqual(Job('2'), population.get_next_job())
        self.assertEqual(Job('3'), population.get_next_job())

    def test_get_next_job_with_features(self):
        population = InfinitePopulation(
            categorical_feature_names=['a', 'b', 'c'],
            numerical_feature_names=['n1', 'n2'],
            seed=42)
        job = population.get_next_job()
        self.assertEqual('0', job.job_id)
        self.assertTrue('a' in job.features)
        self.assertTrue('b' in job.features)
        self.assertTrue('c' in job.features)
        self.assertTrue('n1' in job.features)
        self.assertTrue('n2' in job.features)

if __name__ == '__main__':
    unittest.main()

