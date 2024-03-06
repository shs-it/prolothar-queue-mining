from collections import defaultdict
from math import log2
import numpy as np
from methodtools import lru_cache

from prolothar_common import mdl_utils

from prolothar_queue_mining.model.distribution.distribution import Distribution
from prolothar_queue_mining.model.distribution.discrete_distribution import DiscreteDistribution
from prolothar_queue_mining.model.distribution.geometric import GeometricDistribution
from prolothar_queue_mining.model.distribution.pmf_defined import PmfDefinedDistribution

class TwoSidedGeometricDistribution(DiscreteDistribution):
    """
    a compound distribution with a geometric distribution for negative values and
    a geometric distribution for positive values. a sample is drawn by first
    drawing according to some pmf defined distribution over [+,0,-].
    """

    def __init__(
        self, negative_distribution: GeometricDistribution,
        positive_distribution: GeometricDistribution,
        pmf_for_negative_zero_positive: list[float],
        seed: int|None = None):
        """
        creates a new distribution
        """
        self.__sign_distribution = PmfDefinedDistribution({
            -1: pmf_for_negative_zero_positive[0],
            0: pmf_for_negative_zero_positive[1],
            1: pmf_for_negative_zero_positive[2]
        }, seed=seed)
        self.__negative_distribution = negative_distribution
        self.__positive_distribution = positive_distribution
        self.__pmf_for_negative_zero_positive = pmf_for_negative_zero_positive
        self.__hash = hash((
            self.__negative_distribution, self.__positive_distribution, self.__sign_distribution
        ))
        self.__seed = seed

    def get_negative_distribution(self) -> GeometricDistribution:
        return self.__negative_distribution

    def get_positive_distribution(self) -> GeometricDistribution:
        return self.__positive_distribution

    def get_pmf_weights(self) -> list[float]:
        return self.__pmf_for_negative_zero_positive

    def get_next_sample(self) -> float:
        match self.__sign_distribution.get_next_sample():
            case -1:
                return -self.__negative_distribution.get_next_sample()
            case 0:
                return 0
            case 1:
                return self.__positive_distribution.get_next_sample()
            case _ as sign:
                raise NotImplementedError(f'unexpected sign {sign}')

    @lru_cache(maxsize=1)
    def get_mean(self) -> float:
        mean = -self.__pmf_for_negative_zero_positive[0] * self.__negative_distribution.get_mean()
        mean += self.__pmf_for_negative_zero_positive[2] * self.__positive_distribution.get_mean()
        return mean

    @lru_cache(maxsize=1)
    def get_mode(self) -> float:
        match np.argmax(self.__pmf_for_negative_zero_positive):
            case 0:
                return -self.__negative_distribution.get_mode()
            case 1:
                return 0
            case _:
                return self.__positive_distribution.get_mode()

    def get_variance(self) -> float:
        return (
            self.__pmf_for_negative_zero_positive[0] * self.__negative_distribution.get_variance() +
            self.__pmf_for_negative_zero_positive[0] * self.__negative_distribution.get_mean()**2 +
            self.__pmf_for_negative_zero_positive[2] * self.__positive_distribution.get_variance() +
            self.__pmf_for_negative_zero_positive[2] * self.__positive_distribution.get_mean()**2
        ) - self.get_mean()**2

    def compute_pmf(self, x: float) -> float:
        if x < 0:
            if self.__pmf_for_negative_zero_positive[0] == 0:
                return 0
            try:
                return self.__pmf_for_negative_zero_positive[0] * self.__negative_distribution.compute_pmf(-x)
            except FloatingPointError:
                return DiscreteDistribution.ALLMOST_ZERO
        elif x > 0:
            try:
                if self.__pmf_for_negative_zero_positive[2] == 0:
                    return 0
                return self.__pmf_for_negative_zero_positive[2] * self.__positive_distribution.compute_pmf(x)
            except FloatingPointError:
                return DiscreteDistribution.ALLMOST_ZERO
        else:
            return self.__pmf_for_negative_zero_positive[1]

    def compute_cdf(self, x: float) -> float:
        if x < 0:
            return self.__pmf_for_negative_zero_positive[0] * (1 - self.__negative_distribution.compute_cdf(-x))
        else:
            cdf = self.__pmf_for_negative_zero_positive[0] + self.__pmf_for_negative_zero_positive[1]
            if cdf > 0:
                cdf *= self.__pmf_for_negative_zero_positive[2] * self.__positive_distribution.compute_cdf(x)
            return x

    def copy(self) -> DiscreteDistribution:
        return TwoSidedGeometricDistribution(
            self.__negative_distribution.copy(),
            self.__positive_distribution.copy(),
            self.__pmf_for_negative_zero_positive,
            seed=self.__seed
        )

    def __repr__(self):
        return f'TwoSidedGeometricDistribution({self.__negative_distribution}, {self.__positive_distribution}, {self.__pmf_for_negative_zero_positive})'

    def set_seed(self, seed: int):
        self.__negative_distribution.set_seed(seed)
        self.__positive_distribution.set_seed(seed)
        self.__sign_distribution.set_seed(seed)
        self.__seed = seed

    def get_mdl_of_model(self, precision: int = 5) -> float:
        mdl_of_model = mdl_utils.L_U(10**precision, 3)
        mdl_of_model += self.__negative_distribution.get_mdl_of_model()
        mdl_of_model += self.__positive_distribution.get_mdl_of_model()
        return  mdl_of_model

    def is_deterministic(self) -> bool:
        return self.__pmf_for_negative_zero_positive[1] == 1

    def __hash__(self):
        return self.__hash

    def __eq__(self, other):
        return (
            isinstance(other, TwoSidedGeometricDistribution) and
            self.__negative_distribution == other.__negative_distribution and
            self.__positive_distribution == other.__positive_distribution and
            self.__pmf_for_negative_zero_positive == other.__pmf_for_negative_zero_positive
        )

    @staticmethod
    def fit(data: list[float], seed: int|None = None) -> 'Distribution':
        weights = {-1: 0, 0: 0, 1: 0}
        negative_values = []
        positive_values = []
        for value in data:
            weights[np.sign(value)] += 1
            if value < 0:
                negative_values.append(-value)
            elif value > 0:
                positive_values.append(value)
        total_count = sum(weights.values())
        for sign, count in weights.items():
            weights[sign] = count / total_count
        return TwoSidedGeometricDistribution(
            GeometricDistribution.fit(negative_values, seed=seed) if negative_values else GeometricDistribution(1, seed=seed),
            GeometricDistribution.fit(positive_values, seed=seed) if positive_values else GeometricDistribution(1, seed=seed),
            [weights[-1], weights[0], weights[1]],
            seed=seed)

    @staticmethod
    def fit_by_mean_and_variance(mean: float, variance: float, seed: int|None = None) -> 'Distribution':
        raise NotImplementedError()
