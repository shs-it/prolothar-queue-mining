from abc import ABC, abstractmethod
from prolothar_queue_mining.model.job import Job

class Population(ABC):
    """
    population of jobs (e.g. customers)
    """

    @abstractmethod
    def get_next_job(self) -> Job:
        """
        returns the next job if available or raises a StopIteration error
        """

    @abstractmethod
    def copy(self) -> 'Population':
        """
        returns a deep copy of this population with the same state.
        if an attribute of this population uses a random seed and the
        random seed is "None", this will result in different behavior during
        simulation. this is not a bug, but the reason for this method.
        otherwise use the "deepcopy" python module.
        """

    @abstractmethod
    def set_seed(self, seed: int):
        """
        sets the seed for any random generators used in this population. if
        no randomness is included in this population, the action does not have
        an effect
        """
