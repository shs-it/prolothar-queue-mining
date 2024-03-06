from abc import ABC, abstractmethod
from prolothar_queue_mining.model.job import Job

class Router(ABC):
    """
    determines how a job flows through the network of queues
    """

    @abstractmethod
    def get_name_of_next_queue(self, job: Job, current_time: int) -> str:
        """
        returns the name of the queue to which this job is supposed to go next.
        if the job has been completed, a StopIteration is raised.
        """
