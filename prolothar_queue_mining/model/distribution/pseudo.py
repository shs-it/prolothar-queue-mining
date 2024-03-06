from collections import defaultdict
from prolothar_common.experiments.statistics import Statistics

from prolothar_queue_mining.model.distribution.distribution import Distribution

class PseudoDistribution(Distribution):
    """
    a distribution, that outputs a fixed sequence of values
    """

    def __init__(self, sequence: list[float], current_index: int = 0):
        self.__sequence = sequence
        self.__current_index = current_index
        self.__statistics = Statistics(sequence)
        self.__pdf = defaultdict(float)
        for value in sequence:
            self.__pdf[value] += 1
        for value, count in self.__pdf.items():
            self.__pdf[value] = count / len(sequence)
        self.__mode = max(self.__pdf, key=self.__pdf.get)

        cdf = 0
        self.__cdf = {}
        for value, pdf in sorted(self.__pdf.items()):
            cdf += pdf
            self.__cdf[value] = cdf

        self.__sequence_length = len(sequence)

    def get_next_sample(self) -> float:
        if self.__current_index == self.__sequence_length:
            self.__current_index = 0
        next_sample = self.__sequence[self.__current_index]
        self.__current_index += 1
        return next_sample

    def get_mean(self) -> float:
        return self.__statistics.mean()

    def get_mode(self) -> float:
        return self.__mode

    def get_variance(self) -> float:
        return self.__statistics.variance(degrees_of_freedom=0)

    def copy(self) -> Distribution:
        return PseudoDistribution(self.__sequence, current_index=self.__current_index)

    def __repr__(self):
        return 'PseudoDistribution(...)'

    def compute_pdf(self, x: float) -> float:
        return self.__pdf[x]

    def compute_cdf(self, x: float) -> float:
        try:
            return self.__cdf[x]
        except KeyError as e:
            if x < self.__statistics.minimum():
                return 0
            elif x > self.__statistics.maximum():
                return 1
            else:
                raise NotImplementedError() from e

    def set_seed(self, seed: int):
        #pseudo distribution by definition does not contain randomness
        pass

    def __hash__(self):
        return hash(tuple(self.__sequence))

    def __eq__(self, other):
        return (
            isinstance(other, PseudoDistribution) and
            self.__sequence == other.__sequence
        )

    @staticmethod
    def fit(data: list[float], seed: int|None = None) -> 'Distribution':
        return PseudoDistribution(data)

    @staticmethod
    def fit_by_mean_and_variance(mean: float, variance: float, seed: int|None = None) -> 'Distribution':
        raise NotImplementedError()
