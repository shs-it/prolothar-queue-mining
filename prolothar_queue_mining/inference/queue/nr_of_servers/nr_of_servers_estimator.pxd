from typing import List, Tuple
from prolothar_queue_mining.model.job import Job

cdef class NrOfServersEstimator:

    cpdef int estimate_nr_of_servers(
        self, list observed_arrivals: List[Tuple[Job, float]],
        list observed_departures: List[Tuple[Job, float]])