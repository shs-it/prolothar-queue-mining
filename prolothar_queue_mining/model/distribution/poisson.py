from prolothar_common.experiments.statistics import Statistics
from numpy.random import default_rng
import scipy.stats as stats
from methodtools import lru_cache

from prolothar_common import mdl_utils

from prolothar_queue_mining.model.distribution.distribution import Distribution
from prolothar_queue_mining.model.distribution.discrete_distribution import DiscreteDistribution

class PoissonDistribution(DiscreteDistribution):
    """
    a Poisson distribution
    """

    def __init__(
        self, expected_value: float, seed: int = None, shift: float = 0,
        nr_of_buffered_samples: int = 1000):
        """
        creates a new Poisson distribution with a given expected value (lambda)
        """
        self.expected_value = expected_value
        self.seed = seed
        self.__nr_of_buffered_samples = nr_of_buffered_samples
        self.set_seed(seed)
        self.__shift = shift

    def get_shift(self) -> float:
        return self.__shift

    def get_next_sample(self) -> float:
        try:
            return next(self.__sample_buffer_iterator)
        except StopIteration:
            self.__sample_buffer_iterator =iter(self.__random_generator.poisson(
                self.expected_value, size=self.__nr_of_buffered_samples) + self.__shift)
            return next(self.__sample_buffer_iterator)

    def get_mean(self) -> float:
        return self.expected_value + self.__shift

    def get_mode(self) -> float:
        return int(self.expected_value + self.__shift)

    def get_variance(self) -> float:
        return self.expected_value

    @lru_cache()
    def compute_pmf(self, x: float) -> float:
        if x < self.__shift or (isinstance(x, float) and not x.is_integer()):
            return 0
        try:
            return stats.poisson._pmf(x - self.__shift, self.expected_value)
        except (FloatingPointError, OverflowError):
            return DiscreteDistribution.ALLMOST_ZERO

    @lru_cache()
    def compute_cdf(self, x: float) -> float:
        return stats.poisson.cdf(x - self.__shift, self.expected_value)

    def copy(self) -> DiscreteDistribution:
        return PoissonDistribution(self.expected_value, seed=self.seed, shift=self.__shift)

    def __repr__(self):
        return f'PoissonDistribution({self.expected_value}, shift={self.__shift}, seed={self.seed})'

    def set_seed(self, seed: int|None):
        self.seed = seed
        self.__random_generator = default_rng(seed)
        self.__sample_buffer_iterator = iter([])

    def get_mdl_of_model(self) -> float:
        return mdl_utils.L_R(self.expected_value)

    def is_deterministic(self) -> bool:
        return self.compute_pmf(self.get_mean()) >= 1

    def __hash__(self):
        return hash((self.expected_value, self.__shift))

    def __eq__(self, other):
        return (
            isinstance(other, PoissonDistribution) and
            self.expected_value == other.expected_value and
            self.__shift == other.__shift
        )

    @staticmethod
    def fit(data: list[float], seed: int|None = None) -> 'Distribution':
        data_statistics = Statistics(data)
        return PoissonDistribution(
            data_statistics.mean() - data_statistics.minimum(),
            seed=seed, shift=data_statistics.minimum())

    @staticmethod
    def fit_by_mean_and_variance(mean: float, variance: float, seed: int|None = None) -> 'Distribution':
        shift = mean - variance
        return PoissonDistribution(mean - shift, shift=shift, seed=seed)
