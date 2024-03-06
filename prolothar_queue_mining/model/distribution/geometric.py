import math
from numpy.random import default_rng
import scipy.stats as stats
from prolothar_common.experiments.statistics import Statistics
from methodtools import lru_cache

from prolothar_queue_mining.model.distribution.distribution import Distribution
from prolothar_queue_mining.model.distribution.discrete_distribution import DiscreteDistribution

class GeometricDistribution(DiscreteDistribution):
    """
    a geometric distribution
    https://en.wikipedia.org/wiki/Geometric_distribution
    """

    def __init__(
        self, success_probability: float, seed: int = None,
        nr_of_buffered_samples: int = 1000):
        """
        creates a new geometric distribution with a given success probability
        """
        if not 0 < success_probability <= 1:
            raise ValueError(f'success probability must be in (0,1] but was {success_probability}')
        self.__success_probability = success_probability
        self.__nr_of_buffered_samples = nr_of_buffered_samples
        self.set_seed(seed)

    def get_success_probability(self) -> float:
        return self.__success_probability

    def get_next_sample(self) -> float:
        try:
            return next(self.__sample_buffer_iterator)
        except StopIteration:
            self.__sample_buffer_iterator = iter(self.__random_generator.geometric(
                self.__success_probability, size=self.__nr_of_buffered_samples))
            return next(self.__sample_buffer_iterator)

    def get_mean(self) -> float:
        return 1 / self.__success_probability

    def get_mode(self) -> float:
        return 1

    def get_variance(self) -> float:
        return (1 - self.__success_probability) / self.__success_probability**2

    @lru_cache()
    def compute_pmf(self, x: float) -> float:
        if x < 1:
            return 0
        try:
            return math.pow(1 - self.__success_probability, x - 1) * self.__success_probability
        except (FloatingPointError, OverflowError):
            #x is way too large to compute the geometric PMF value, which goes
            #towards 0 for large x
            return DiscreteDistribution.ALLMOST_ZERO

    @lru_cache()
    def compute_cdf(self, x: float) -> float:
        return stats.geom.cdf(x, self.__success_probability)

    def copy(self) -> DiscreteDistribution:
        return GeometricDistribution(self.__success_probability, seed=self.__seed)

    def __repr__(self):
        return f'GeometricDistribution({self.__success_probability}, seed={self.__seed})'

    def set_seed(self, seed: int|None):
        self.__seed = seed
        self.__random_generator = default_rng(seed)
        self.__sample_buffer_iterator = iter([])

    def get_mdl_of_model(self, precision: int = 5) -> float:
        return precision * math.log2(10)

    def is_deterministic(self) -> bool:
        return False

    def __hash__(self):
        return hash(self.__success_probability)

    def __eq__(self, other):
        return (
            isinstance(other, GeometricDistribution) and
            self.__success_probability == other.__success_probability
        )

    @staticmethod
    def fit(data: list[float], seed: int|None = None) -> DiscreteDistribution:
        statistics = Statistics(data)
        return GeometricDistribution.fit_by_mean_and_variance(statistics.mean(), float('nan'))

    @staticmethod
    def fit_by_mean_and_variance(mean: float, variance: float, seed: int|None = None) -> 'Distribution':
        return GeometricDistribution(min(1, 1 / mean), seed=seed)

