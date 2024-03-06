from abc import ABC, abstractmethod
from copy import deepcopy

from prolothar_queue_mining.model.job import Job

class SojournTimeObserver(ABC):
    """
    interface of an observer for sojourn times, i.e. the time between arrival and
    exit
    """

    @abstractmethod
    def notify(self, job: Job, arrival_time: int, exit_time: int):
        """
        method to notify that a job has exited

        Parameters
        ----------
        job : Job
            job that has exited the queue
        arrival_time : int
            original time of arrival for this job
        exit_time : int
            time when the job has left the queue
        """

    def copy(self) -> 'SojournTimeObserver':
        """
        returns a deep copy of this sojourn time observer with the same state
        """
        return deepcopy(self)
