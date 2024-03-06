from abc import ABC, abstractmethod

from copy import deepcopy
from prolothar_queue_mining.model.job import Job

class WaitingTimeObserver(ABC):
    """
    interface of an observer for waiting times, i.e. the time between arrival and
    start of service
    """

    @abstractmethod
    def notify(self, job: Job, arrival_time: int, start_of_service_time: int):
        """
        method to notify that serving of a new job has been started

        Parameters
        ----------
        job : Job
            job for which serving has been started
        arrival_time : float
            original time of arrival for this job
        start_of_service_time : float
            current time at which processing has been start. the difference between
            start_of_service_time and arrival_time is the waiting_time.
        """

    def copy(self) -> 'WaitingTimeObserver':
        """
        returns a deep copy of this waiting time observer with the same state
        """
        return deepcopy(self)
