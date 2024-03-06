from collections import deque

from prolothar_queue_mining.model.arrival_process.arrival_process import ArrivalProcess
from prolothar_queue_mining.model.distribution import Distribution
from prolothar_queue_mining.model.job import Job

class BatchArrivalAdapter(ArrivalProcess):
    """
    translates any other arrival process into a batch arrival process. the arrival time
    of each job in a batch is defined as the arrival time of the latest job in the
    batch.
    """

    def __init__(
            self, individual_arrival_process: ArrivalProcess,
            batch_size_distribution: Distribution):
        self.__individual_arrival_process = individual_arrival_process
        self.__batch_size_distribution = batch_size_distribution
        self.__open_jobs = deque()
        self.__current_arrival_time = 0

    def get_next_job(self) -> tuple[int,Job]:
        #refill open jobs => create a new batch
        if not self.__open_jobs:
            for _ in range(max(1, self.__batch_size_distribution.get_next_sample())):
                self.__current_arrival_time, job = self.__individual_arrival_process.get_next_job()
                self.__open_jobs.append(job)
        return self.__current_arrival_time, self.__open_jobs.popleft()

    def get_mean_arrival_rate(self) -> float:
        return self.__individual_arrival_process.get_mean_arrival_rate()

    def copy(self) -> ArrivalProcess:
        return BatchArrivalAdapter(
            self.__individual_arrival_process.copy(),
            self.__batch_size_distribution.copy())

    def set_seed(self, seed: int):
        self.__individual_arrival_process.set_seed(seed)
        self.__batch_size_distribution.set_seed(seed)

    def __repr__(self):
        return f'BatchArrivalAdapter({self.__individual_arrival_process}, {self.__batch_size_distribution})'
