from math import exp, log as ln
import numpy as np
from scipy.stats import lognorm
from numpy.random import default_rng

from prolothar_common import mdl_utils

from prolothar_queue_mining.model.distribution.distribution import Distribution
from prolothar_queue_mining.model.distribution.continuous_distribution import ContinuousDistribution

class LogNormalDistribution(ContinuousDistribution):
    """
    a log normal distribution
    """

    def __init__(self, mu: float, sigma: float, seed: int|None = None):
        """
        creates a new log normal distribution with the given mean and standard deviation
        https://en.wikipedia.org/wiki/Log-normal_distribution

        Parameters
        ----------
        mu : float
        sigma : float
        seed : float, optional
            random generator seed, by default None
        """
        self.__mu = mu
        self.__sigma = sigma
        self.__scale = exp(self.__mu)
        self.__sigma_square = self.__sigma * self.__sigma
        self.set_seed(seed)

    def get_next_sample(self) -> float:
        return self.__random_generator.lognormal(self.__mu, self.__sigma)

    def get_mean(self) -> float:
        return exp(self.__mu + self.__sigma_square / 2)

    def get_mode(self) -> float:
        return exp(self.__mu - self.__sigma_square)

    def get_variance(self) -> float:
        return (exp(self.__sigma_square) - 1) * exp(2 * self.__mu + self.__sigma_square)

    def compute_pdf(self, x: float) -> float:
        return lognorm.pdf(x, self.__sigma, scale=self.__scale)

    def compute_cdf(self, x: float) -> float:
        return lognorm.cdf(x, self.__sigma, scale=self.__scale)

    def get_mdl_of_model(self) -> float:
        return mdl_utils.L_R(self.__mu) + mdl_utils.L_R(self.__sigma)

    def copy(self) -> ContinuousDistribution:
        return LogNormalDistribution(self.__mu, self.__sigma, seed=self.seed)

    def __repr__(self):
        return f'LogNormalDistribution({self.__mu}, {self.__sigma}, seed={self.seed})'

    def set_seed(self, seed: int):
        self.seed = seed
        self.__random_generator = default_rng(seed)

    def __hash__(self):
        return hash((self.__mu, self.__sigma))

    def __eq__(self, other):
        return (
            isinstance(other, LogNormalDistribution) and
            self.__mu == other.__mu and
            self.__sigma == other.__sigma
        )

    @staticmethod
    def fit(data: list[float], seed: int|None = None) -> 'Distribution':
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
        sigma, _, scale = lognorm.fit(data, loc=0)
        return LogNormalDistribution(ln(scale), sigma, seed=seed)

    @staticmethod
    def fit_by_mean_and_variance(mean: float, variance: float, seed: int|None = None) -> 'Distribution':
        raise NotImplementedError()