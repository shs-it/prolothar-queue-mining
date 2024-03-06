from math import log2
import numpy as np
from numpy.random import default_rng
import scipy.stats as stats
from distfit import distfit
from methodtools import lru_cache

from prolothar_common import mdl_utils

from prolothar_queue_mining.model.distribution.distribution import Distribution
from prolothar_queue_mining.model.distribution.discrete_distribution import DiscreteDistribution

class BinomialDistribution(DiscreteDistribution):
    """
    a binomial distribution
    https://en.wikipedia.org/wiki/Binomial_distribution
    """

    def __init__(self, nr_of_trials: float, success_probability: float, seed: int|None = None):
        """
        creates a new binomial distribution with a given number of
        successes until experiment is stopped and a success probability in each
        experiment
        """
        if not 0 < success_probability < 1:
            raise ValueError(f'success probability must be in [0,1] but was {success_probability}')
        if nr_of_trials <= 0:
            raise ValueError(f'nr_of_trials must not be <= 0')
        self.__success_probability = success_probability
        self.__nr_of_trials = nr_of_trials
        self.set_seed(seed)

    def get_next_sample(self) -> float:
        return self.__random_generator.binomial(
            self.__nr_of_trials, self.__success_probability)

    def get_mean(self) -> float:
        return self.__nr_of_trials * self.__success_probability

    def get_mode(self) -> float:
        return round(self.__nr_of_trials * self.__success_probability)

    def get_variance(self) -> float:
        return self.__nr_of_trials * (1 - self.__success_probability) * self.__success_probability

    @lru_cache()
    def compute_pmf(self, x: float) -> float:
        if x < 0 or (isinstance(x, float) and not x.is_integer()):
            return 0
        try:
            pmf = stats.binom._pmf(x, self.__nr_of_trials, self.__success_probability)
            return 0 if np.isnan(pmf) else pmf
        except (FloatingPointError, OverflowError):
            #x is way too large to compute the NB PMF value, which goes
            #towards 0 for large x
            return DiscreteDistribution.ALLMOST_ZERO

    @lru_cache()
    def compute_cdf(self, x: float) -> float:
        return stats.binom.cdf(x, self.__nr_of_trials, self.__success_probability)

    def copy(self) -> DiscreteDistribution:
        return BinomialDistribution(
            self.__nr_of_trials, self.__success_probability,
            seed=self.__seed)

    def __repr__(self):
        return (
            f'BinomialDistribution({self.__nr_of_trials}, '
            f'{self.__success_probability}, '
            f'seed={self.__seed})'
        )

    def set_seed(self, seed: int):
        self.__seed = seed
        self.__random_generator = default_rng(seed)

    def get_mdl_of_model(self, precision: int = 5) -> float:
        return precision * log2(10) + mdl_utils.L_R(self.__nr_of_trials)

    def is_deterministic(self) -> bool:
        return False

    def __hash__(self):
        return hash((self.__success_probability, self.__nr_of_trials))

    def __eq__(self, other):
        return (
            isinstance(other, BinomialDistribution) and
            self.__success_probability == other.__success_probability and
            self.__nr_of_trials == other.__nr_of_trials
        )

    @staticmethod
    def fit(data: list[float], seed: int|None = None) -> 'Distribution':
        dist = distfit(method='discrete')
        model = dist.fit_transform(np.array(data), verbose=0)['model']
        if model['name'] != 'binom':
            raise NotImplementedError(model)
        return BinomialDistribution(model['n'], model['p'], seed=seed)

    @staticmethod
    def fit_by_mean_and_variance(mean: float, variance: float, seed: int|None = None) -> 'Distribution':
        success_probability = 1 - variance / mean
        if success_probability <= 0:
            raise NotImplementedError()
        nr_of_trials = mean / success_probability
        return BinomialDistribution(nr_of_trials, success_probability, seed=seed)
