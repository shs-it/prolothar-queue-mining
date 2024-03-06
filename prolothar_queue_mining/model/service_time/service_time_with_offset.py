from prolothar_queue_mining.model.service_time.service_time import ServiceTime
from prolothar_queue_mining.model.job import Job

class ServiceTimeWithOffset(ServiceTime):
    """
    a meta service time, that adds an offet to the prediction of its child
    """

    def __init__(self, service_time: ServiceTime, offset: int):
        self.__service_time = service_time
        self.__offset = offset

    def get_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int:
        return self.__service_time.get_service_time(job, nr_of_jobs_in_system) + self.__offset

    def get_expected_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int:
        return self.__service_time.get_expected_service_time(job, nr_of_jobs_in_system) + self.__offset

    def get_most_likely_service_time(self, job: Job, nr_of_jobs_in_system: int) -> tuple[int, float]:
        time, probability = self.__service_time.get_most_likely_service_time(job, nr_of_jobs_in_system)
        time += self.__offset
        return time, probability

    def get_batch_service_time(self, batch: list[Job], nr_of_jobs_in_system: int) -> int:
        return self.__service_time.get_batch_service_time(batch, nr_of_jobs_in_system) + self.__offset

    def copy(self) -> ServiceTime:
        return ServiceTimeWithOffset(self.__service_time.copy(), self.__offset)

    def compute_probability(self, x: int, job: Job, nr_of_jobs_in_system: int) -> float:
        return self.__service_time.compute_probability(x - self.__offset, job, nr_of_jobs_in_system)

    def compute_max_probability(self, x: int) -> float:
        return self.__service_time.compute_max_probability(x - self.__offset)

    def set_seed(self, seed: int):
        self.__service_time.set_seed(seed)

    def is_deterministic(self) -> bool:
        return self.__service_time.is_deterministic()

    def __repr__(self):
        return f'{self.__service_time} + {self.__offset}'
