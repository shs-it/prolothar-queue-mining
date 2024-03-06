import os
import unittest
import tempfile

from prolothar_queue_mining.model.distribution import GeometricDistribution

from prolothar_queue_mining.model.distribution.plots import plot_pmf
from prolothar_queue_mining.model.distribution.plots import plot_pmfs

class TestDistributionPlots(unittest.TestCase):

    def test_plot_pmf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_file = os.path.join(temp_dir, 'pmf.png')
            self.assertFalse(os.path.exists(plot_file))
            plot_pmf(GeometricDistribution(0.3), range(10), show=False, filepath=plot_file)
            self.assertTrue(os.path.exists(plot_file))

    def test_plot_pmfs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_file = os.path.join(temp_dir, 'pmfs.png')
            self.assertFalse(os.path.exists(plot_file))
            plot_pmfs(
                [GeometricDistribution(0.3), GeometricDistribution(0.1)],
                range(10), show=False, filepath=plot_file)
            self.assertTrue(os.path.exists(plot_file))

if __name__ == '__main__':
    unittest.main()