from scipy.stats import gamma
import numpy as np
from numpy.random import default_rng
from methodtools import lru_cache

from prolothar_queue_mining.model.distribution.distribution import Distribution
from prolothar_queue_mining.model.distribution.continuous_distribution import ContinuousDistribution

class GammaDistribution(ContinuousDistribution):
    """
    a Gamma distribution
    https://en.wikipedia.org/wiki/Gamma_distribution
    """

    def __init__(self, shape: float, rate: float, seed: int|None = None):
        """
        creates a new gamma distribution with the given shape and rate

        Parameters
        ----------
        shape : float
            alpha in literature
        rate : float
            beta in literature
        seed : float, optional
            random generator seed, by default None
        """
        if shape <= 0:
            raise ValueError(f'shape must not be <= 0, but was {shape}')
        if rate <= 0:
            raise ValueError(f'rate must not be <= 0, but was {rate}')
        self.shape = shape
        self.rate = rate
        self.__scale = 1 / rate
        self.set_seed(seed)

    def get_next_sample(self) -> float:
        return self.__random_generator.gamma(self.shape, self.__scale)

    def get_mean(self) -> float:
        return self.shape / self.rate

    def get_mode(self) -> float:
        return (max(1, self.shape) - 1) * self.__scale

    def get_variance(self) -> float:
        return self.shape / (self.rate)**2

    def copy(self) -> ContinuousDistribution:
        return GammaDistribution(self.shape, self.rate, seed=self.seed)

    @lru_cache()
    def compute_pdf(self, x: float) -> float:
        return gamma.pdf(x, self.shape, 0, self.__scale)

    @lru_cache()
    def compute_cdf(self, x: float) -> float:
        return gamma.cdf(x, self.shape, 0, self.__scale)

    def __repr__(self):
        return f'GammaDistribution({self.shape}, {self.rate}, seed={self.seed})'

    def set_seed(self, seed: int):
        self.seed = seed
        self.__random_generator = default_rng(seed)

    def __hash__(self):
        return hash((self.shape, self.rate))

    def __eq__(self, other):
        return (
            isinstance(other, GammaDistribution) and
            self.shape == other.shape and
            self.rate == other.rate
        )

    @staticmethod
    def fit(data: list[float], seed: int|None = None) -> 'Distribution':
        alpha, _, beta = gamma.fit(data, floc=0)
        if np.isnan(alpha) or np.isnan(beta):
            raise ValueError(f'alpha={alpha}, beta={beta}')
        return GammaDistribution(alpha, 1/beta, seed=seed)

    @staticmethod
    def fit_by_mean_and_variance(mean: float, variance: float, seed: int|None = None) -> 'Distribution':
        rate = mean / variance
        shape = mean * rate
        return GammaDistribution(shape, rate, seed=seed)