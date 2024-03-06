from abc import ABC, abstractmethod

from prolothar_queue_mining.model.job import Job

class Exit(ABC):
    """
    an exit point of a queue that handles jobs leaving the queue
    """

    @abstractmethod
    def add_job(self, timestep: int, job: Job):
        """
        handle job arrival at a given timestep at this exit

        Parameters
        ----------
        timestep : int
            timestep the job arrives at this exit point
        job : object
            job that arrives at this exit point
        """

    def copy(self) -> 'Exit':
        """
        returns a deep copy of this exit time with the same state.
        if an attribute of this exit time uses a random seed and the
        random seed is "None", this will result in different behavior during
        simulation. this is not a bug, but the reason for this method.
        otherwise use the "deepcopy" python module.
        """
