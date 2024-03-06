from math import log2
import numpy as np
from numpy.random import default_rng
import scipy.stats as stats
import statsmodels.api as sm
from methodtools import lru_cache

from prolothar_common import mdl_utils

from prolothar_queue_mining.model.distribution.distribution import Distribution
from prolothar_queue_mining.model.distribution.discrete_distribution import DiscreteDistribution

class NegativeBinomialDistribution(DiscreteDistribution):
    """
    a negative binomial distribution
    https://en.wikipedia.org/wiki/Negative_binomial_distribution
    """

    def __init__(
        self, nr_of_successes: float, success_probability: float, seed: int|None = None,
        nr_of_buffered_samples: int = 1000):
        """
        creates a new negative binomial distribution with a given number of
        successes until experiment is stopped and a success probability in each
        experiment
        """
        if not 0 < success_probability < 1:
            raise ValueError(f'success probability must be in [0,1] but was {success_probability}')
        if nr_of_successes <= 0:
            raise ValueError(f'nr_of_successes must not be <= 0')
        self.__success_probability = success_probability
        self.__nr_of_successes = nr_of_successes
        self.__nr_of_buffered_samples = nr_of_buffered_samples
        self.set_seed(seed)

    def get_next_sample(self) -> float:
        try:
            return next(self.__sample_buffer_iterator)
        except StopIteration:
            self.__sample_buffer_iterator = iter(self.__random_generator.negative_binomial(
                self.__nr_of_successes, self.__success_probability, size=self.__nr_of_buffered_samples))
            return next(self.__sample_buffer_iterator)

    def get_mean(self) -> float:
        return self.__nr_of_successes * (1 - self.__success_probability) / self.__success_probability

    def get_mode(self) -> float:
        if self.__nr_of_successes <= 1:
            return 0
        return int((1 - self.__success_probability) * (self.__nr_of_successes - 1) / self.__success_probability)

    def get_variance(self) -> float:
        return self.__nr_of_successes * (1 - self.__success_probability) / self.__success_probability**2

    def is_deterministic(self) -> bool:
        return False

    @lru_cache()
    def compute_pmf(self, x: float) -> float:
        if x < 0 or (isinstance(x, float) and not x.is_integer()):
            return 0
        try:
            return stats.nbinom._pmf(x, self.__nr_of_successes, self.__success_probability)
        except (FloatingPointError, OverflowError):
            #x is way too large to compute the NB PMF value, which goes
            #towards 0 for large x
            return DiscreteDistribution.ALLMOST_ZERO

    @lru_cache()
    def compute_cdf(self, x: float) -> float:
        return stats.nbinom.cdf(x, self.__nr_of_successes, self.__success_probability)

    def copy(self) -> DiscreteDistribution:
        return NegativeBinomialDistribution(
            self.__nr_of_successes, self.__success_probability,
            seed=self.__seed)

    def __repr__(self):
        return (
            f'NegativeBinomialDistribution({self.__nr_of_successes}, '
            f'{self.__success_probability}, '
            f'seed={self.__seed})'
        )

    def set_seed(self, seed: int|None):
        self.__seed = seed
        self.__random_generator = default_rng(seed)
        self.__sample_buffer_iterator = iter([])

    def get_mdl_of_model(self, precision: int = 5) -> float:
        try:
            return precision * log2(10) + mdl_utils.L_R(self.__nr_of_successes)
        except OverflowError:
            return precision * log2(10) + mdl_utils.L_R(2**16)

    def __hash__(self):
        return hash((self.__nr_of_successes, self.__success_probability))

    def __eq__(self, other):
        return (
            isinstance(other, NegativeBinomialDistribution) and
            self.__nr_of_successes == other.__nr_of_successes and
            self.__success_probability == other.__success_probability
        )

    @staticmethod
    def fit(data: list[float], seed: int|None = None) -> 'Distribution':
        mle_result = sm.NegativeBinomial(data, np.ones_like(data)).fit(start_params=[1, 1], disp=0)
        estimated_mean = np.exp(mle_result.params[0])
        estimated_nr_of_successes = 1 / mle_result.params[1]
        estimated_success_probability = 1 / (estimated_mean / estimated_nr_of_successes + 1)
        return NegativeBinomialDistribution(estimated_nr_of_successes, estimated_success_probability, seed=seed)

    @staticmethod
    def fit_by_mean_and_variance(mean: float, variance: float, seed: int|None = None) -> 'Distribution':
        success_probability = mean / variance
        nr_of_successes = (mean * success_probability) / (1 - success_probability)
        return NegativeBinomialDistribution(nr_of_successes, success_probability, seed=seed)
