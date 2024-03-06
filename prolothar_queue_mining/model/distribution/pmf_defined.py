from collections import defaultdict
from math import log2
import scipy.stats as stats

from prolothar_common import mdl_utils

from prolothar_queue_mining.model.distribution.distribution import Distribution
from prolothar_queue_mining.model.distribution.discrete_distribution import DiscreteDistribution

class PmfDefinedDistribution(DiscreteDistribution):
    """
    an arbitrary distribution that is purely defined by its PMF
    """

    def __init__(self, pmf: dict[int, float], seed: int|None = None):
        """
        creates a new distribution with a given PMF
        """
        if not (0.999 < sum(pmf.values()) < 1.0001):
            raise ValueError(f'pmf must sum up to 1 but sums up to {sum(pmf.values())}')
        x, px = zip(*list(pmf.items()))
        self.__rv_discrete = stats.rv_discrete(values=(x, px))
        self.__mean = 0
        self.__max_pmf = 0
        self.__mode = None
        self.__pmf = pmf
        for x, px in pmf.items():
            self.__mean += px * x
            if px > self.__max_pmf:
                self.__mode = x
        self.__hash = hash(tuple(self.__pmf.items()))
        self.set_seed(seed)
        self.__sample_buffer_iterator = iter([])

    def get_next_sample(self) -> float:
        try:
            return next(self.__sample_buffer_iterator)
        except StopIteration:
            self.__sample_buffer_iterator = iter(self.__rv_discrete.rvs(size=1000))
            return next(self.__sample_buffer_iterator)

    def get_mean(self) -> float:
        return self.__mean

    def get_mode(self) -> float:
        return self.__mode

    def get_variance(self) -> float:
        return self.__rv_discrete.var()

    def compute_pmf(self, x: float) -> float:
        try:
            return self.__pmf[x]
        except KeyError:
            return 0

    def compute_cdf(self, x: float) -> float:
        return self.__rv_discrete.cdf(x)

    def copy(self) -> DiscreteDistribution:
        return PmfDefinedDistribution(self.__pmf, seed=self.__seed)

    def __repr__(self):
        sorted_items = sorted(self.__pmf.items())
        if len(sorted_items) > 10:
            sorted_items = sorted_items[:3] + ['...'] + sorted_items[-3:]
        return (
            f'PmfDefinedDistribution({sorted_items}, '
            f'seed={self.__seed})'
        )

    def set_seed(self, seed: int):
        self.__seed = seed

    def get_mdl_of_model(self, precision: int = 5) -> float:
        min_x = min(self.__pmf)
        max_x = max(self.__pmf)
        mdl_of_model = mdl_utils.L_N(min_x+1) + mdl_utils.L_N(max_x - min_x + 1)
        if max_x > min_x:
            mdl_of_model += log2(max_x - min_x)
            mdl_of_model += mdl_utils.log2binom(max_x - min_x, len(self.__pmf))
            mdl_of_model += mdl_utils.L_U(10**precision, len(self.__pmf))
        return mdl_of_model

    def is_deterministic(self) -> bool:
        return len(self.__pmf) > 1

    def __hash__(self):
        return self.__hash

    def __eq__(self, other):
        return (
            isinstance(other, PmfDefinedDistribution) and
            self.__pmf == other.__pmf
        )

    @staticmethod
    def fit(data: list[float], seed: int|None = None, laplace_smoothing: bool = False) -> 'Distribution':
        counter = defaultdict(int)
        for value in data:
            counter[value] += 1
        if laplace_smoothing:
            for i in range(min(counter), max(counter) + 1):
                counter[i] += 1
        total_count = sum(counter.values())
        for value, count in counter.items():
            counter[value] = count / total_count
        return PmfDefinedDistribution(dict(counter), seed=seed)

    @staticmethod
    def fit_by_mean_and_variance(mean: float, variance: float, seed: int|None = None) -> 'Distribution':
        raise NotImplementedError()
