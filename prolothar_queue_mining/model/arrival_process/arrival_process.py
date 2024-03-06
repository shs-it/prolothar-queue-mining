from abc import ABC, abstractmethod

from prolothar_queue_mining.model.job import Job

class ArrivalProcess(ABC):
    """
    Arrival process of the queue that determines which jobs arrive at which time
    """

    @abstractmethod
    def get_next_job(self) -> tuple[int,Job]:
        """
        returns the next job that arrives due to this arrival process together
        with the time.
        returns time, job
        """

    @abstractmethod
    def copy(self) -> 'ArrivalProcess':
        """
        returns a deep copy of this arrival process with the same state.
        if an attribute of this arrival process uses a random seed and the
        random seed is "None", this will result in different behavior during
        simulation. this is not a bug, but the reason for this method.
        otherwise use the "deepcopy" python module.
        """

    @abstractmethod
    def set_seed(self, seed: int):
        """
        sets the seed used in random generators in this arrival process
        """

    @abstractmethod
    def get_mean_arrival_rate(self) -> float:
        """
        returns the mean or expected arrival rate, i.e. the number of jobs
        arriving at one timestep (usually a fractional).
        """
