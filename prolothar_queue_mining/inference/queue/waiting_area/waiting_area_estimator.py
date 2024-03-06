from abc import ABC, abstractmethod

from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.job import Job

class WaitingAreaEstimator(ABC):
    """
    interface of a waiting area estimator, i.e. an algorithm that infers the structure
    of the waiting area of a queue from observations. in our case, only arrival and departure
    times are given.
    """

    @abstractmethod
    def infer_waiting_area(
        self, observed_arrivals: list[tuple[Job, int]],
        observed_departures: list[tuple[Job, int]]) -> WaitingArea|None:
        """
        tries to infer structure and parameters of a WaitingArea model from a list
        of observed job arrivals and departues

        Parameters
        ----------
        observed_arrivals : list[tuple[Job, int]]
            a list of jobs together with the observed arrival time, ordered
            by arrival time (smaller arrival times at the beginning)
        observed_departures : list[tuple[Job, int]]
            a list of jobs together with the observed departure time, ordered
            by departure time (smaller departure times at the beginning)

        Returns
        -------
        WaitingArea|None
            a waiting area model inferred from observed_arrivals and observed_departues.
            can be None if the estimator contains some criterion that denies fit
            on the given data.
        """
