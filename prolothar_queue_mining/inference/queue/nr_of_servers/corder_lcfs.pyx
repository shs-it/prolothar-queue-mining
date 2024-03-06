from typing import List, Tuple

from prolothar_queue_mining.model.job import Job

from prolothar_queue_mining.inference.queue.nr_of_servers.nr_of_servers_estimator import NrOfServersEstimator
from prolothar_queue_mining.inference.queue.nr_of_servers.utils import create_combined_job_index_list

from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set

class COrderLcfs(NrOfServersEstimator):
    """
    c_order_lcfs in original implementation.
    estimates the number of servers in a LCFS queue using the order-based algorithm.

    Andrew Keith and Darryl Ahner and Raymond Hill
    "An order-based method for robust queue inference with stochastic arrival and departure times"
    Computers & Industrial Engineering
    2019

    originally implemented in
    https://github.com/ajkeith/UnobservableQueue.jl
    """

    def estimate_nr_of_servers(
            self, observed_arrivals: List[Tuple[Job, int]],
            observed_departures: List[Tuple[Job, int]]) -> int:
        cdef vector[int] combined_order = create_combined_job_index_list(observed_arrivals, observed_departures)
        cdef int nr_of_servers = 0
        cdef unordered_set[int] jobs_in_queue = unordered_set[int]()
        cdef unordered_set[int] jobs_in_service = unordered_set[int]()
        cdef int job
        for job in combined_order:
            if jobs_in_queue.find(job) != jobs_in_queue.end():
                jobs_in_queue.erase(job)
                jobs_in_service.erase(job)
                if jobs_in_queue.size() > 0 and jobs_in_queue.size() > jobs_in_service.size():
                    #add the highest job not in service
                    jobs_in_service.insert(get_highest_job_not_in_service(jobs_in_queue, jobs_in_service))
            else:
                jobs_in_queue.insert(job)
            nr_of_servers = max(nr_of_servers, jobs_in_service.size())
        return max(nr_of_servers, 1)

cdef int get_highest_job_not_in_service(unordered_set[int]& jobs_in_queue, unordered_set[int]& jobs_in_service):
    cdef int highest_job = 0
    cdef int job
    for job in jobs_in_queue:
        if jobs_in_service.find(job) == jobs_in_service.end() and job > highest_job:
            highest_job = job
    return highest_job
