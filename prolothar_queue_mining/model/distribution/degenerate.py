from prolothar_common.experiments.statistics import Statistics

from prolothar_common import mdl_utils

from prolothar_queue_mining.model.distribution.distribution import Distribution
from prolothar_queue_mining.model.distribution.discrete_distribution import DiscreteDistribution
from prolothar_queue_mining.model.distribution.continuous_distribution import ContinuousDistribution

class DiscreteDegenerateDistribution(DiscreteDistribution):
    """
    a distribution, that can only output a single number
    """

    def __init__(self, value: float):
        self.value = value

    def get_next_sample(self) -> float:
        return self.value

    def get_mean(self) -> float:
        return self.value

    def get_mode(self) -> float:
        return self.value

    def get_variance(self) -> float:
        return 0

    def copy(self) -> DiscreteDistribution:
        return DiscreteDegenerateDistribution(self.value)

    def __repr__(self):
        return f'DegenerateDistribution({self.value})'

    def compute_pmf(self, x: float) -> float:
        return 1 if self.value == x else 0

    def compute_cdf(self, x: float) -> float:
        return 0 if x < self.value else 1

    def set_seed(self, seed: int):
        #degenerate distribution does not have randomness by definition
        pass

    def get_mdl_of_model(self) -> float:
        return mdl_utils.L_R(self.value)

    def is_deterministic(self) -> bool:
        return True

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return (
            isinstance(other, DiscreteDegenerateDistribution) and
            self.value == other.value
        )

    @staticmethod
    def fit(data: list[float], seed: int|None = None) -> 'Distribution':
        return DiscreteDegenerateDistribution(round(Statistics(data).mean()))

    @staticmethod
    def fit_by_mean_and_variance(mean: float, variance: float, seed: int|None = None) -> 'Distribution':
        return DiscreteDegenerateDistribution(round(mean))

class ContinuousDegenerateDistribution(ContinuousDistribution):
    """
    a distribution, that can only output a single number
    """

    def __init__(self, value: float):
        self.value = value

    def get_next_sample(self) -> float:
        return self.value

    def get_mean(self) -> float:
        return self.value

    def get_mode(self) -> float:
        return self.value

    def get_variance(self) -> float:
        return 0

    def copy(self) -> ContinuousDistribution:
        return ContinuousDegenerateDistribution(self.value)

    def __repr__(self):
        return f'DegenerateDistribution({self.value})'

    def compute_pdf(self, x: float) -> float:
        return float('inf') if self.value == x else 0

    def compute_cdf(self, x: float) -> float:
        return 0 if x < self.value else float('inf')

    def set_seed(self, seed: int):
        #degenerate distribution does not have randomness by definition
        pass

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return (
            isinstance(other, ContinuousDegenerateDistribution) and
            self.value == other.value
        )

    @staticmethod
    def fit(data: list[float], seed: int|None = None) -> 'Distribution':
        return ContinuousDegenerateDistribution(Statistics(data).mean())

    @staticmethod
    def fit_by_mean_and_variance(mean: float, variance: float, seed: int|None = None) -> 'Distribution':
        return ContinuousDegenerateDistribution(mean)
