from prolothar_queue_mining.model.service_time.service_time import ServiceTime
from prolothar_queue_mining.model.job import Job

class FixedServiceTime(ServiceTime):
    """
    a fixed, i.e. constant, service time
    """

    def __init__(self, service_time: int):
        self.__service_time = service_time

    def get_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int:
        return self.__service_time

    def get_expected_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int:
        return self.__service_time

    def get_most_likely_service_time(self, job: Job, nr_of_jobs_in_system: int) -> tuple[int, float]:
        return self.__service_time, 1

    def get_batch_service_time(self, batch: list[Job], nr_of_jobs_in_system: int) -> int:
        return self.__service_time

    def copy(self) -> ServiceTime:
        return FixedServiceTime(self.__service_time)

    def compute_probability(self, x: int, job: Job, nr_of_jobs_in_system: int) -> float:
        return self.compute_max_probability(x)

    def compute_max_probability(self, x: int) -> float:
        if x == self.__service_time:
            return 1
        else:
            return 0

    def set_seed(self, seed: int):
        #no randomness included
        pass

    def is_deterministic(self) -> bool:
        return True

    def __repr__(self):
        return f'Fixed({self.__service_time})'
