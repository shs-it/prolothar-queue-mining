import unittest
import numpy as np
from prolothar_common.experiments.statistics import Statistics

from prolothar_queue_mining.model.distribution import Distribution
from prolothar_queue_mining.model.distribution import DiscreteDegenerateDistribution
from prolothar_queue_mining.model.distribution import ContinuousDegenerateDistribution
from prolothar_queue_mining.model.distribution import NormalDistribution
from prolothar_queue_mining.model.distribution import ExponentialDistribution
from prolothar_queue_mining.model.distribution import PoissonDistribution
from prolothar_queue_mining.model.distribution import GammaDistribution
from prolothar_queue_mining.model.distribution import PseudoDistribution
from prolothar_queue_mining.model.distribution import NegativeBinomialDistribution
from prolothar_queue_mining.model.distribution import GeometricDistribution
from prolothar_queue_mining.model.distribution import BinomialDistribution
from prolothar_queue_mining.model.distribution import PmfDefinedDistribution
from prolothar_queue_mining.model.distribution import C2dDistribution
from prolothar_queue_mining.model.distribution import LogNormalDistribution
from prolothar_queue_mining.model.distribution import GaussianMixtureModelDistribution
from prolothar_queue_mining.model.distribution import TwoSidedGeometricDistribution

class TestDistribution(unittest.TestCase):

    def test_discrete_degenerate_distribution(self):
        self.assertEqual(2, DiscreteDegenerateDistribution(2).get_next_sample())
        self.assertEqual(3, DiscreteDegenerateDistribution(3).get_next_sample())
        self.assertEqual(4, DiscreteDegenerateDistribution(4).get_next_sample())
        self.assertEqual(4, DiscreteDegenerateDistribution(4).get_mean())
        self.assertEqual(4, DiscreteDegenerateDistribution(4).get_mode())
        self.assertEqual(0, DiscreteDegenerateDistribution(4).get_variance())

        self._test_fit(DiscreteDegenerateDistribution(4))
        self._test_hashcode_and_eq(DiscreteDegenerateDistribution(4))

        self.assertEqual(0, DiscreteDegenerateDistribution(4).compute_pmf(3.9))
        self.assertEqual(1, DiscreteDegenerateDistribution(4).compute_pmf(4))

        self.assertEqual(1, DiscreteDegenerateDistribution(4).compute_cdf(4))
        self.assertEqual(0, DiscreteDegenerateDistribution(4).compute_cdf(3.9))

    def test_continuous_degenerate_distribution(self):
        self.assertEqual(2, ContinuousDegenerateDistribution(2).get_next_sample())
        self.assertEqual(3, ContinuousDegenerateDistribution(3).get_next_sample())
        self.assertEqual(4, ContinuousDegenerateDistribution(4).get_next_sample())
        self.assertEqual(4, ContinuousDegenerateDistribution(4).get_mean())
        self.assertEqual(4, ContinuousDegenerateDistribution(4).get_mode())
        self.assertEqual(0, ContinuousDegenerateDistribution(4).get_variance())

        self._test_fit(ContinuousDegenerateDistribution(4))

        self.assertEqual(0, ContinuousDegenerateDistribution(4).compute_pdf(3.9))
        self.assertEqual(float('inf'), ContinuousDegenerateDistribution(4).compute_pdf(4))

        self.assertEqual(float('inf'), ContinuousDegenerateDistribution(4).compute_cdf(4))
        self.assertEqual(0, ContinuousDegenerateDistribution(4).compute_cdf(3.9))

    def test_normal_distribution(self):
        distribution = NormalDistribution(0, 1, seed=42)
        statistics = Statistics([
            distribution.get_next_sample() for _ in range(1000)
        ])
        self.assertAlmostEqual(0, statistics.mean(), delta=0.03)
        self.assertAlmostEqual(1, statistics.stddev(), delta=0.03)
        self.assertEqual(0, distribution.get_mean())
        self.assertEqual(0, distribution.get_mode())
        self.assertEqual(1, distribution.get_variance())

        self._test_fit(distribution)
        self._test_hashcode_and_eq(distribution)

        self.assertAlmostEqual(0.399, distribution.compute_pdf(0), delta=0.001)
        self.assertAlmostEqual(0.5, distribution.compute_cdf(0), delta=0.001)
        self.assertAlmostEqual(0.841, distribution.compute_cdf(1), delta=0.001)

    def test_gaussian_mixture_model_distribution(self):
        distribution = GaussianMixtureModelDistribution([0.8, 0.2], [1, 4], [2, 1], seed=4022022)
        statistics = Statistics([
            distribution.get_next_sample() for _ in range(20000)
        ])
        self.assertAlmostEqual(distribution.get_mean(), statistics.mean(), delta=0.03)
        self.assertAlmostEqual(distribution.get_variance(), statistics.variance(), delta=0.1)
        self.assertEqual(1, distribution.get_mode())

        self._test_fit(distribution, nr_of_samples=10000)
        self._test_hashcode_and_eq(distribution)

        self.assertAlmostEqual(0.176, distribution.compute_pdf(0), delta=0.001)

    def test_exponential_distribution(self):
        distribution = ExponentialDistribution(0.5, seed=42)
        statistics = Statistics([
            distribution.get_next_sample() for _ in range(10000)
        ])
        self.assertAlmostEqual(2, statistics.mean(), delta=0.03)
        self.assertEqual(2, distribution.get_mean())
        self.assertEqual(0, distribution.get_mode())
        self.assertAlmostEqual(statistics.variance(), distribution.get_variance(), delta=0.1)

        self._test_fit(distribution)
        self._test_hashcode_and_eq(distribution)

        self.assertAlmostEqual(0.184, distribution.compute_pdf(2), delta=0.001)
        self.assertAlmostEqual(0.303, distribution.compute_pdf(1), delta=0.001)
        self.assertAlmostEqual(0.5, distribution.compute_pdf(0), delta=0.001)

        self.assertAlmostEqual(0.0, distribution.compute_cdf(0), delta=0.001)
        self.assertAlmostEqual(0.393, distribution.compute_cdf(1), delta=0.001)

    def test_poisson_distribution(self):
        distribution = PoissonDistribution(2, seed=42)
        statistics = Statistics([
            distribution.get_next_sample() for _ in range(1000)
        ])
        self.assertAlmostEqual(2, statistics.mean(), delta=0.03)
        self.assertAlmostEqual(2, statistics.variance(), delta=0.03)
        self.assertEqual(2, distribution.get_mean())
        self.assertEqual(2, distribution.get_mode())
        self.assertEqual(2, distribution.get_variance())

        self._test_fit(distribution)
        self._test_fit(PoissonDistribution(2, shift=1, seed=23))
        self._test_hashcode_and_eq(PoissonDistribution(2, shift=1, seed=23))

        self.assertAlmostEqual(0.175, PoissonDistribution(5).compute_pmf(4), delta=0.001)
        self.assertAlmostEqual(0.0, PoissonDistribution(5).compute_pmf(4.5), delta=0.001)

        self.assertAlmostEqual(0.677, distribution.compute_cdf(2), delta=0.001)

    def test_gamma_distribution(self):
        distribution = GammaDistribution(2, 1/2, seed=42)
        statistics = Statistics([
            distribution.get_next_sample() for _ in range(2000)
        ])
        self.assertAlmostEqual(4, statistics.mean(), delta=0.1)
        self.assertEqual(4, distribution.get_mean())
        self.assertEqual(2, distribution.get_mode())
        self.assertAlmostEqual(statistics.variance(), distribution.get_variance(), delta=0.2)
        self.assertAlmostEqual(6.5, GammaDistribution(7.5, 1).get_mode(), delta=0.001)

        self._test_fit(GammaDistribution(2, 1/3, seed=42), nr_of_samples=1000)
        self._test_hashcode_and_eq(GammaDistribution(2, 1/3, seed=42))

        self.assertAlmostEqual(0.0, GammaDistribution(2, 2).compute_pdf(0), delta=0.001)
        self.assertAlmostEqual(0.541, GammaDistribution(2, 2).compute_pdf(1), delta=0.001)
        self.assertAlmostEqual(0.368, GammaDistribution(2, 1).compute_pdf(1), delta=0.001)

        self.assertAlmostEqual(0.393, GammaDistribution(1, 0.5).compute_cdf(1), delta=0.001)

    def test_lognormal_distribution(self):
        distribution = LogNormalDistribution(0, 1, seed=42)
        statistics = Statistics([
            distribution.get_next_sample() for _ in range(10000)
        ])
        self.assertAlmostEqual(distribution.get_mean(), statistics.mean(), delta=0.1)
        self.assertAlmostEqual(0.35, distribution.get_mode(), delta=0.1)
        self.assertAlmostEqual(statistics.variance(), distribution.get_variance(), delta=0.2)

        self._test_fit(LogNormalDistribution(2, 1/3, seed=42), nr_of_samples=10000)
        self._test_hashcode_and_eq(LogNormalDistribution(2, 1/3, seed=42))

        self.assertAlmostEqual(0.0, LogNormalDistribution(0, 1).compute_pdf(0), delta=0.001)
        self.assertAlmostEqual(0.399, LogNormalDistribution(0, 1).compute_pdf(1), delta=0.001)
        self.assertAlmostEqual(0.5, LogNormalDistribution(0, 1).compute_cdf(1), delta=0.001)

    def test_geometric_distribution(self):
        distribution = GeometricDistribution(1/2, seed=42)
        statistics = Statistics([
            distribution.get_next_sample() for _ in range(1000)
        ])
        self.assertAlmostEqual(statistics.mean(), distribution.get_mean(), delta=0.1)
        self.assertAlmostEqual(statistics.variance(), distribution.get_variance(), delta=0.1)
        self.assertEqual(1, distribution.get_mode())

        self._test_fit(GeometricDistribution(1/2, seed=42))
        self._test_fit(GeometricDistribution(1/3, seed=42))
        self._test_hashcode_and_eq(GeometricDistribution(1/3, seed=42))

        self.assertAlmostEqual(0.5, GeometricDistribution(1/2).compute_pmf(1), delta=0.01)
        self.assertAlmostEqual(0.75, GeometricDistribution(1/2).compute_cdf(2), delta=0.01)

    def test_negative_binomial_distribution(self):
        distribution = NegativeBinomialDistribution(1, 1/2, seed=42)
        statistics = Statistics([
            distribution.get_next_sample() for _ in range(2000)
        ])
        self.assertAlmostEqual(1, statistics.mean(), delta=0.1)
        self.assertEqual(1, distribution.get_mean())
        self.assertEqual(0, distribution.get_mode())
        self.assertAlmostEqual(statistics.variance(), distribution.get_variance(), delta=0.2)

        distribution = NegativeBinomialDistribution(2, 1/3, seed=42)
        statistics = Statistics([
            distribution.get_next_sample() for _ in range(2000)
        ])
        self.assertAlmostEqual(statistics.mean(), distribution.get_mean(), delta=0.1)
        self._test_fit(NegativeBinomialDistribution(1, 1/2, seed=42))
        self._test_fit(NegativeBinomialDistribution(2, 1/3, seed=42), nr_of_samples=2000)
        self._test_fit(NegativeBinomialDistribution(4, 1/4, seed=42), nr_of_samples=2000)
        self._test_hashcode_and_eq(NegativeBinomialDistribution(4, 1/4, seed=42))

        self.assertAlmostEqual(0.111, NegativeBinomialDistribution(2, 1/3).compute_pmf(0), delta=0.01)
        self.assertAlmostEqual(0.148, NegativeBinomialDistribution(2, 1/3).compute_pmf(2), delta=0.01)
        self.assertAlmostEqual(0.064, NegativeBinomialDistribution(4, 0.25).compute_pmf(9), delta=0.01)

        self.assertAlmostEqual(0.595, NegativeBinomialDistribution(4, 0.25).compute_cdf(12), delta=0.01)

        distribution = NegativeBinomialDistribution(1.263385811041596, 0.0039305316227113175, seed=None)
        self.assertGreater(distribution.compute_pmf(distribution.get_mode()), 0)
        self.assertAlmostEqual(distribution.compute_pmf(distribution.get_mode()), 0.002, delta=0.01)
        self.assertGreater(distribution.compute_pmf(1000), 0)

    def test_binomial_distribution(self):
        distribution = BinomialDistribution(20, 0.7, seed=42)
        statistics = Statistics([
            distribution.get_next_sample() for _ in range(2000)
        ])
        self.assertAlmostEqual(distribution.get_mean(), statistics.mean(), delta=0.1)
        self.assertEqual(14, distribution.get_mean())
        self.assertEqual(14, distribution.get_mode())
        self.assertAlmostEqual(statistics.variance(), distribution.get_variance(), delta=0.2)

        self._test_fit(BinomialDistribution(1, 1/2, seed=42))
        self._test_fit(BinomialDistribution(2, 1/3, seed=42))
        self._test_fit(BinomialDistribution(4, 1/4, seed=42))
        self._test_hashcode_and_eq(BinomialDistribution(4, 1/4, seed=42))

        self.assertAlmostEqual(0.2, BinomialDistribution(20, 0.7).compute_pmf(14), delta=0.01)

    def test_pmf_defined_distribution(self):
        distribution = PmfDefinedDistribution({
            0: 0.3,
            1: 0.2,
            2: 0.5
        }, seed=42)
        statistics = Statistics([
            distribution.get_next_sample() for _ in range(2000)
        ])
        self.assertAlmostEqual(distribution.get_mean(), statistics.mean(), delta=0.1)
        self.assertEqual(2, distribution.get_mode())
        self.assertAlmostEqual(statistics.variance(), distribution.get_variance(), delta=0.2)

        self._test_fit(distribution)
        self._test_hashcode_and_eq(distribution)

        self.assertEqual(0.3, distribution.compute_pmf(0))
        self.assertEqual(0.2, distribution.compute_pmf(1))
        self.assertEqual(0.5, distribution.compute_pmf(2))
        self.assertEqual(0.0, distribution.compute_pmf(3))
        self.assertEqual(0.0, distribution.compute_pmf(-1))

    def test_c2d_distribution(self):
        distribution = C2dDistribution(NormalDistribution(1.5, 3, seed=42))
        statistics = Statistics([
            distribution.get_next_sample() for _ in range(100000)
        ])
        self.assertAlmostEqual(distribution.get_mean(), statistics.mean(), delta=0.1)
        self.assertEqual(2, distribution.get_mode())
        self.assertAlmostEqual(statistics.variance(), distribution.get_variance(), delta=0.2)

        self._test_hashcode_and_eq(distribution)

        self.assertAlmostEqual(0.131, distribution.compute_pmf(1), delta=0.01)

    def test_pseudo_distribution(self):
        distribution = PseudoDistribution([1,2,3,4,5,6,7,8,9,10])
        statistics = Statistics([
            distribution.get_next_sample() for _ in range(1000)
        ])
        self.assertAlmostEqual(5.5, statistics.mean(), delta=0.1)
        self.assertEqual(5.5, distribution.get_mean())
        self.assertAlmostEqual(statistics.variance(), distribution.get_variance(), delta=0.01)
        self.assertEqual(2, PseudoDistribution([1,2,2]).get_mode())

        self._test_fit(PseudoDistribution([1,2,3,4,5,6,7,8,9,10]))
        self._test_hashcode_and_eq(PseudoDistribution([1,2,3,4,5,6,7,8,9,10]))

        self.assertAlmostEqual(0.0, distribution.compute_pdf(0), delta=0.001)
        self.assertAlmostEqual(0.1, distribution.compute_pdf(1), delta=0.001)
        self.assertAlmostEqual(0.1, distribution.compute_pdf(2), delta=0.001)
        self.assertAlmostEqual(2/3, PseudoDistribution([1,2,2]).compute_pdf(2), delta=0.001)

        self.assertAlmostEqual(0.1, distribution.compute_cdf(1), delta=0.001)
        self.assertAlmostEqual(0.2, distribution.compute_cdf(2), delta=0.001)
        self.assertAlmostEqual(1.0, distribution.compute_cdf(10), delta=0.001)
        self.assertAlmostEqual(1.0, distribution.compute_cdf(11), delta=0.001)

    def test_two_sided_geometric_distribution(self):
        distribution = TwoSidedGeometricDistribution(
            GeometricDistribution(0.3, seed=25032022),
            GeometricDistribution(0.3, seed=25032022),
            [1/3, 1/3, 1/3], seed=25032022
        )
        self.assertAlmostEqual(0, distribution.get_mean())
        statistics = Statistics([
            distribution.get_next_sample() for _ in range(1000000)
        ])
        self.assertAlmostEqual(distribution.get_mean(), statistics.mean(), delta=0.1)
        self.assertAlmostEqual(distribution.get_variance(), statistics.variance(), delta=0.25)

        self._test_fit(distribution, nr_of_samples=100000)
        self._test_fit(TwoSidedGeometricDistribution(
            GeometricDistribution(0.3, seed=25032022),
            GeometricDistribution(0.2, seed=25032022),
            [0.3, 0.5, 0.2], seed=25032022
        ), nr_of_samples=100000)
        self._test_hashcode_and_eq(distribution)
        self.assertAlmostEqual(1/3, distribution.compute_pmf(0))

    def _test_fit(self, distribution: Distribution, nr_of_samples: int = 1000, seed: int = 42):
        values = [distribution.get_next_sample() for _ in range(nr_of_samples)]
        fitted_distribution = distribution.fit(values, seed=seed)
        self.assertAlmostEqual(distribution.get_mean(), fitted_distribution.get_mean(), delta=0.2)
        self.assertIsInstance(fitted_distribution, type(distribution))

        if not isinstance(distribution, PseudoDistribution) \
        and not isinstance(distribution, PmfDefinedDistribution) \
        and not isinstance(distribution, GaussianMixtureModelDistribution) \
        and not isinstance(distribution, TwoSidedGeometricDistribution) \
        and not isinstance(distribution, LogNormalDistribution):
            fitted_distribution = distribution.fit_by_mean_and_variance(
                distribution.get_mean(), distribution.get_variance(), seed=seed)
            self.assertAlmostEqual(distribution.get_mean(), fitted_distribution.get_mean(), delta=0.0001)
            self.assertAlmostEqual(distribution.get_variance(), fitted_distribution.get_variance(), delta=0.0001)

    def _test_hashcode_and_eq(self, distribution: Distribution):
        copy = distribution.copy()
        self.assertIsNot(copy, distribution)
        self.assertEqual(hash(distribution), hash(copy))
        self.assertEqual(distribution, copy)

if __name__ == '__main__':
    unittest.main()