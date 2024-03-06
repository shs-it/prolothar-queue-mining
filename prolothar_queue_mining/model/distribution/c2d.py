from methodtools import lru_cache

from prolothar_queue_mining.model.distribution.distribution import Distribution
from prolothar_queue_mining.model.distribution.discrete_distribution import DiscreteDistribution
from prolothar_queue_mining.model.distribution.continuous_distribution import ContinuousDistribution

class C2dDistribution(DiscreteDistribution):
    """
    a wrapper distribution, that transforms a continuous distribution to a
    discrete distribution
    """

    def __init__(self, wrapped_distribution: ContinuousDistribution):
        self.__wrapper_distribution = wrapped_distribution

    def get_next_sample(self) -> float:
        return round(self.__wrapper_distribution.get_next_sample())

    def get_mean(self) -> float:
        return self.__wrapper_distribution.get_mean()

    def get_mode(self) -> float:
        return round(self.__wrapper_distribution.get_mode())

    def get_variance(self) -> float:
        return self.__wrapper_distribution.get_variance()

    def copy(self) -> DiscreteDistribution:
        return C2dDistribution(self.__wrapper_distribution.copy())

    def __repr__(self):
        return f'C2d({self.__wrapper_distribution})'

    @lru_cache()
    def compute_pmf(self, x: float) -> float:
        return self.__wrapper_distribution.compute_cdf(x + 0.5) - self.__wrapper_distribution.compute_cdf(x - 0.5)

    @lru_cache()
    def compute_cdf(self, x: float) -> float:
        return self.__wrapper_distribution.compute_cdf(x)

    def set_seed(self, seed: int):
        self.__wrapper_distribution.set_seed(seed)

    def get_mdl_of_model(self) -> float:
        return self.__wrapper_distribution.get_mdl_of_model()

    def __hash__(self):
        return hash(self.__wrapper_distribution)

    def __eq__(self, other):
        return (
            isinstance(other, C2dDistribution) and
            self.__wrapper_distribution == other.__wrapper_distribution
        )

    def is_deterministic(self) -> bool:
        return self.compute_pmf(self.get_mean()) < 1

    @staticmethod
    def fit(data: list[float], seed: int|None = None) -> 'Distribution':
        raise NotImplementedError()

    @staticmethod
    def fit_by_mean_and_variance(mean: float, variance: float, seed: int|None = None) -> 'Distribution':
        raise NotImplementedError()

