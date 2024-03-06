from math import log2
from prolothar_queue_mining.model.service_time.service_time import ServiceTime
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.distribution import DiscreteDistribution
from prolothar_queue_mining.model.distribution import DiscreteDegenerateDistribution

class ServiceTimeWithDistribution(ServiceTime):
    """
    a service time that follows any distribution. to get valid results,
    returned service times are truncated to positive values and round to integer.
    """

    def __init__(self, distribution: DiscreteDistribution):
        self.__distribution = distribution

    def get_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int:
        return max(0, round(self.__distribution.get_next_sample()))

    def get_batch_service_time(self, batch: list[Job], nr_of_jobs_in_system: int) -> int:
        return max(0, round(self.__distribution.get_next_sample()))

    def copy(self) -> ServiceTime:
        #for efficiency resons, we do not copy the distribution, that has
        #cached PMF and CDF values
        return ServiceTimeWithDistribution(self.__distribution)

    def copy_mean(self) -> 'ServiceTime':
        return ServiceTimeWithDistribution(DiscreteDegenerateDistribution(self.__distribution.get_mean()))

    def compute_probability(self, x: int, job: Job, nr_of_jobs_in_system: int) -> float:
        return self.__distribution.compute_pmf(x)

    def compute_max_probability(self, x: int) -> float:
        return self.__distribution.compute_pmf(x)

    def get_expected_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int:
        return round(self.__distribution.get_mean())

    def get_most_likely_service_time(self, job: Job, nr_of_jobs_in_system: int) -> tuple[int, float]:
        return self.__distribution.get_mode(), self.__distribution.compute_pmf(self.__distribution.get_mode())

    def get_distribution(self) -> DiscreteDistribution:
        return self.__distribution

    def set_seed(self, seed: int):
        self.__distribution.set_seed(seed)

    def get_mdl_of_model(self) -> float:
        return self.__distribution.get_mdl_of_model()

    def get_min_code_length_for_one_job(self) -> float:
        return -log2(self.__distribution.compute_pmf(self.__distribution.get_mode()))

    def is_deterministic(self) -> bool:
        return self.__distribution.is_deterministic()

    def __repr__(self):
        return f'ServiceTime({self.__distribution})'
