from prolothar_queue_mining.model.arrival_process.arrival_process import ArrivalProcess
from prolothar_queue_mining.model.job import Job

class LimitedArrival(ArrivalProcess):
    """
    limits the maximal number of arrived jobs by any other arrival process
    """

    def __init__(self, individual_arrival_process: ArrivalProcess, max_nr_of_jobs: int):
        self.__individual_arrival_process = individual_arrival_process
        self.__nr_of_jobs = 0
        self.__max_nr_of_jobs = max_nr_of_jobs

    def get_next_job(self) -> tuple[int,Job]:
        if self.__nr_of_jobs == self.__max_nr_of_jobs:
            raise StopIteration()
        self.__nr_of_jobs += 1
        return self.__individual_arrival_process.get_next_job()

    def get_mean_arrival_rate(self) -> float:
        return self.__individual_arrival_process.get_mean_arrival_rate()

    def copy(self) -> ArrivalProcess:
        return LimitedArrival(
            self.__individual_arrival_process.copy(),
            self.__max_nr_of_jobs)

    def set_seed(self, seed: int):
        self.__individual_arrival_process.set_seed(seed)

    def __repr__(self):
        return f'LimitedArrival({self.__individual_arrival_process}, {self.__max_nr_of_jobs})'
