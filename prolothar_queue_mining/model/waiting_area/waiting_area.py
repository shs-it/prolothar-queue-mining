from typing import Iterator
from abc import ABC, abstractmethod

from prolothar_queue_mining.model.job import Job

class WaitingArea(ABC):
    """
    model for the waiting area of the queue. determines the discipline of the queue (e.g. FCFS)
    """

    @abstractmethod
    def add_job(self, arrival_time: int, job: Job):
        """
        adds a job to this waiting area
        """

    @abstractmethod
    def has_next_job(self) -> bool:
        """
        returns True if there is a waiting job, otherwise returns False
        """

    @abstractmethod
    def pop_next_job(self, nr_of_jobs_in_system: int) -> Job:
        """
        returns the next waiting job if there is one. otherwise raises a StopIteration.
        the order in which jobs are returned can be dependent on the nr_of_jobs_in_system,
        i.e. defined as the number of jobs that have arrived but not exited the queue, yet.
        """

    @abstractmethod
    def pop_batch(self, batch_size: int, nr_of_jobs_in_system: int) -> list[Job]:
        """
        returns the next "batch_size" waiting jobs if there are enough jobs waiting.
        otherwise raises a StopIteration.
        the order in which jobs are returned can be dependent on the nr_of_jobs_in_system,
        i.e. defined as the number of jobs that have arrived but not exited the queue, yet.
        """

    @abstractmethod
    def __len__(self):
        """
        returns the number of jobs in this waiting area
        """

    @abstractmethod
    def copy(self) -> 'WaitingArea':
        """
        returns a deep copy of this waiting area with the same state.
        if an attribute of this waiting area uses a random seed and the
        random seed is "None", this will result in different behavior during
        simulation. this is not a bug, but the reason for this method.
        otherwise use the "deepcopy" python module.
        """

    @abstractmethod
    def copy_empty(self) -> 'WaitingArea':
        """
        returns a deep copy of this waiting area. if the waiting area contains jobs,
        the copy will contain no jobs. use copy() if you want to have the same
        content.
        """

    @abstractmethod
    def get_discipline_name(self) -> str:
        """
        returns the name of queue discipline, i.e. "D" in Kendal's notation,
        e.g. FCFS, LCFS, PQ, SIRO, ...
        """

    @abstractmethod
    def any_order_iterator(self) -> Iterator[Job]:
        """
        returns an iterator over the jobs in this waiting area. the jobs
        are not necessarily ordered by their future departure order
        """

    def get_mdl(self, nr_of_categorical_features: int) -> float:
        """
        returns the minimum-description-length in bits to
        encode this waiting area. this is used in CueMin to compute the model complexity.
        """
        return 0

    @abstractmethod
    def get_best_case_sort_key_for_synchronized_arrival(self, job: Job, exit_time: int):
        """
        returns a comparable key that orders jobs with equal arrival time such that the
        number of required servers is minimized
        """

    @abstractmethod
    def get_worst_case_sort_key_for_synchronized_arrival(self, job: Job, exit_time: int):
        """
        returns a comparable key that orders jobs with equal arrival time such that the
        number of required servers is maximized
        """
