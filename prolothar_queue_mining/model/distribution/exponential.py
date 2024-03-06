from numpy.random import default_rng
import scipy.stats as stats
from methodtools import lru_cache

from prolothar_queue_mining.model.distribution.distribution import Distribution
from prolothar_queue_mining.model.distribution.continuous_distribution import ContinuousDistribution

class ExponentialDistribution(ContinuousDistribution):
    """
    an exponential distribution
    """

    def __init__(self, rate: float, seed: int|None = None):
        """
        creates a new exponential distribution with the rate (lambda)
        """
        self.rate = rate
        self.__scale = 1 / rate
        self.set_seed(seed)

    def get_next_sample(self) -> float:
        return self.__random_generator.exponential(self.__scale)

    def get_mean(self) -> float:
        return 1 / self.rate

    def get_mode(self) -> float:
        return 0

    def get_variance(self) -> float:
        return 1 / self.rate**2

    @lru_cache()
    def compute_pdf(self, x: float) -> float:
        return stats.expon.pdf(x, 0, self.__scale)

    @lru_cache()
    def compute_cdf(self, x: float) -> float:
        return stats.expon.cdf(x, 0, self.__scale)

    def copy(self) -> ContinuousDistribution:
        return ExponentialDistribution(self.rate, seed=self.seed)

    def __repr__(self):
        return f'ExponentialDistribution({self.rate}, seed={self.seed})'

    def set_seed(self, seed: int):
        self.seed = seed
        self.__random_generator = default_rng(seed)

    def __hash__(self):
        return hash(self.rate)

    def __eq__(self, other):
        return (
            isinstance(other, ExponentialDistribution) and
            self.rate == other.rate
        )

    @staticmethod
    def fit(data: list[float], seed: int|None = None) -> 'Distribution':
        return ExponentialDistribution(1 / stats.expon.fit(data, floc=0)[1], seed=seed)

    @staticmethod
    def fit_by_mean_and_variance(mean: float, variance: float, seed: int|None = None) -> 'Distribution':
        return ExponentialDistribution(1 / mean, seed=seed)
