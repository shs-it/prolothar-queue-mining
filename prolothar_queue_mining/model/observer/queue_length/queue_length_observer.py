from abc import ABC, abstractmethod
from copy import deepcopy

class QueueLengthObserver(ABC):
    """
    interface of an observer for queue length, i.e. the number of jobs that are
    waiting for service
    """

    @abstractmethod
    def notify(self, current_time: float, queue_length: int):
        """
        method to notify that a job has exited

        Parameters
        ----------
        current_time : float
            current time at which the length of the queue is observed
        queue_length : int
            current length of the queue
        """

    def copy(self) -> 'QueueLengthObserver':
        """
        returns a deep copy of this sojourn time observer with the same state
        """
        return deepcopy(self)
