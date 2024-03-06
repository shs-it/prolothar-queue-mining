from math import sqrt
from scipy.stats import norm
from scipy.special import erf
from numpy.random import default_rng
from methodtools import lru_cache

from prolothar_common import mdl_utils

from prolothar_queue_mining.model.distribution.distribution import Distribution
from prolothar_queue_mining.model.distribution.continuous_distribution import ContinuousDistribution

class NormalDistribution(ContinuousDistribution):
    """
    a normal or Gaussian distribution
    """

    def __init__(
        self, mean: float, stddev: float, seed: int = None,
        nr_of_buffered_samples: int = 1000):
        """
        creates a new normal distribution with the given mean and standard deviation

        Parameters
        ----------
        mean : float
        stddev : float
        seed : float, optional
            random generator seed, by default None
        """
        self.__mean = mean
        self.__stddev = stddev
        self.__variance = stddev**2
        self.__nr_of_buffered_samples = nr_of_buffered_samples
        self.set_seed(seed)

    def get_next_sample(self) -> float:
        try:
            return next(self.__sample_buffer_iterator)
        except StopIteration:
            self.__sample_buffer_iterator = iter(self.__random_generator.normal(
                self.__mean, self.__stddev, size=self.__nr_of_buffered_samples))
            return next(self.__sample_buffer_iterator)

    def get_mean(self) -> float:
        return self.__mean

    def get_mode(self) -> float:
        return self.__mean

    def get_variance(self) -> float:
        return self.__variance

    @lru_cache()
    def compute_pdf(self, x: float) -> float:
        return norm.pdf(x, self.__mean, self.__stddev)

    @lru_cache()
    def compute_cdf(self, x: float) -> float:
        return 0.5 * (1 + _cached_erf((x - self.__mean) / self.__stddev / 1.41421))

    def get_mdl_of_model(self) -> float:
        return mdl_utils.L_R(self.__mean) + mdl_utils.L_R(self.__stddev)

    def copy(self) -> ContinuousDistribution:
        return NormalDistribution(self.__mean, self.__stddev, seed=self.seed)

    def __repr__(self):
        return f'NormalDistribution({self.__mean}, {self.__stddev}, seed={self.seed})'

    def set_seed(self, seed: int|None):
        self.seed = seed
        self.__random_generator = default_rng(seed)
        self.__sample_buffer_iterator = iter([])

    def __hash__(self):
        return hash((self.__mean, self.__stddev))

    def __eq__(self, other):
        return (
            isinstance(other, NormalDistribution) and
            self.__mean == other.__mean and
            self.__variance == other.__variance
        )

    @staticmethod
    def fit(data: list[float], seed: int|None = None) -> 'Distribution':
        mean, stddev = norm.fit(data)
        return NormalDistribution(mean, stddev, seed=seed)

    @staticmethod
    def fit_by_mean_and_variance(mean: float, variance: float, seed: int|None = None) -> 'Distribution':
        return NormalDistribution(mean, sqrt(variance), seed=seed)

@lru_cache()
def _cached_erf(x: float) -> float:
    return erf(x)