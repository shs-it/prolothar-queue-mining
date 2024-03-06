from typing import List, Tuple
from prolothar_queue_mining.model.job import Job

cdef class NrOfServersEstimator():
    """
    interface for the estimation of the number of servers (c in Kendall's notation)
    """

    cpdef int estimate_nr_of_servers(
        self, list observed_arrivals: List[Tuple[Job, int]],
        list observed_departures: List[Tuple[Job, int]]):
        """
        estimates the number of servers in the queue from observed arrival
        and departure of jobs

        Parameters
        ----------
        observed_arrivals : list[tuple[Job, int]]
            a list of jobs and corresponding arrival times
        observed_departures : list[tuple[Job, int]]
            a list of jobs and corresponding departure times

        Returns
        -------
        int
            the number of estimated servers in the queue
        """
        raise NotImplementedError()