from math import log2
from prolothar_queue_mining.model.distribution import DiscreteDistribution

class MdlBatchSizeDistribution():

    def __init__(self, distribution: DiscreteDistribution, observed_batch_sizes: list[int]):
        self.__distribution = distribution
        self.__observed_batch_sizes = observed_batch_sizes
        self.__current_index = 0
        self.__total_encoded_length = 0

    def get_distribution(self) -> DiscreteDistribution:
        return self.__distribution

    def get_mean(self) -> float:
        return self.__distribution.get_mean()

    def get_variance(self) -> float:
        return self.__distribution.get_variance()

    def copy(self) -> 'MdlBatchSizeDistribution':
        return MdlBatchSizeDistribution(self.__distribution.copy(), self.__observed_batch_sizes)

    def get_next_sample(self) -> int:
        if self.__current_index < len(self.__observed_batch_sizes):
            batch_size = self.__observed_batch_sizes[self.__current_index]
            self.__current_index += 1
            probability = self.__distribution.compute_pmf(batch_size)
            if probability == 0:
                batch_size = self.__distribution.get_mode()
                probability = self.__distribution.compute_pmf(batch_size)
            self.__total_encoded_length -= log2(probability)
            return batch_size
        elif self.__current_index == len(self.__observed_batch_sizes):
            self.__current_index += 1
            return self.__distribution.get_mode()
        else:
            batch_size = self.__distribution.get_mode()
            self.__total_encoded_length -= log2(self.__distribution.compute_pmf(batch_size))
            return batch_size

    def get_total_encoded_length(self) -> float:
        return self.__total_encoded_length
