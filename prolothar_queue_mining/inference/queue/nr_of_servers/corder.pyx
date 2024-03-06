from typing import List, Tuple
cimport cython

from prolothar_queue_mining.model.job import Job

from prolothar_queue_mining.inference.queue.nr_of_servers.nr_of_servers_estimator cimport NrOfServersEstimator
from prolothar_queue_mining.inference.queue.nr_of_servers.utils import create_job_departure_order_list

from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set

cdef class COrder(NrOfServersEstimator):
    """
    c_order_slow in original implementation.
    estimates the number of servers in a FCFS queue using the order-based algorithm
    with sets.

    Andrew Keith and Darryl Ahner and Raymond Hill
    "An order-based method for robust queue inference with stochastic arrival and departure times"
    Computers & Industrial Engineering
    2019

    originally implemented in
    https://github.com/ajkeith/UnobservableQueue.jl
    """

    cpdef int estimate_nr_of_servers(
            self, list observed_arrivals: List[Tuple[Job, int]],
            list observed_departures: List[Tuple[Job, int]]):

        cdef vector[int] departure_order = create_job_departure_order_list(observed_arrivals, observed_departures)

        cdef int current_max = 0
        cdef int nr_of_servers = 0
        cdef unordered_set[int] departed_jobs = unordered_set[int]()
        departed_jobs.reserve(departure_order.size())
        cdef int job
        cdef int new_job
        cdef int job2
        cdef int nr_of_jobs_in_service
        for job in departure_order:
            if job > current_max:
                current_max = job
            departed_jobs.insert(job)
            nr_of_jobs_in_service = 0
            for job2 in range(current_max):
                if departed_jobs.find(job2) == departed_jobs.end():
                    nr_of_jobs_in_service += 1
            if nr_of_jobs_in_service > nr_of_servers:
                nr_of_servers = nr_of_jobs_in_service
        return nr_of_servers

