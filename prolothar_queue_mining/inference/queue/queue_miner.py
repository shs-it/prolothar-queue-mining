from abc import ABC, abstractmethod

from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.job import Job

class QueueMiner(ABC):
    """
    interface for a queue miner, i.e. an algorithm that infers the structure
    of a single queue from observations. in our case, only arrival and departure
    times are given. the queue miner is supposed to infer the waiting area
    with its discipline (e.g. LIFO) and the number of servers with the service
    time distribution
    """

    @abstractmethod
    def infer_queue(
        self, observed_arrivals: list[tuple[Job, int]],
        observed_departures: list[tuple[Job, int]]) -> Queue:
        """
        tries to infer structure and parameters of a Queue model from a list
        of observed job arrivals and departues

        Parameters
        ----------
        observed_arrivals : list[tuple[Job, float]]
            a list of jobs together with the observed arrival time, ordered
            by arrival time (smaller arrival times at the beginning)
        observed_departures : list[tuple[Job, float]]
            a list of jobs together with the observed departure time, ordered
            by departure time (smaller departure times at the beginning)

        Returns
        -------
        Queue
            a Queue model inferred from observed_arrivals and observed_departues
        """
