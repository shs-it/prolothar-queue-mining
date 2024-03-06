from numpy.random import default_rng
import scipy.stats as stats

from prolothar_queue_mining.model.service_time.service_time import ServiceTime
from prolothar_queue_mining.model.job import Job

class ExponentialDistributedServiceTime(ServiceTime):

    def __init__(self, mean_service_rate: float, seed: int|None = None):
        self.__mean_service_rate = mean_service_rate
        self.__scale = 1 / self.__mean_service_rate
        self.set_seed(seed)

    def get_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int:
        return round(self.__random_number_generator.exponential(self.__scale))

    def get_expected_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int:
        return round(self.__scale)

    def get_most_likely_service_time(self, job: Job, nr_of_jobs_in_system: int) -> tuple[int, float]:
        return 0, self.compute_probability(0)

    def get_batch_service_time(self, batch: list[Job], nr_of_jobs_in_system: int) -> int:
        return round(self.__random_number_generator.exponential(self.__scale))

    def copy(self) -> ServiceTime:
        return ExponentialDistributedServiceTime(self.__mean_service_rate, seed=self.__seed)

    def compute_probability(self, x: int, job: Job, nr_of_jobs_in_system: int) -> float:
        return self.compute_max_probability(x)

    def compute_max_probability(self, x: int) -> float:
        return stats.expon.cdf(x + 0.5, 0, self.__scale) - stats.expon.cdf(x - 0.5, 0, self.__scale)

    def __repr__(self):
        return f'Exponential({self.__mean_service_rate})'

    def is_deterministic(self) -> bool:
        return False

    def set_seed(self, seed: int):
        self.__random_number_generator = default_rng(seed)
        self.__seed = seed