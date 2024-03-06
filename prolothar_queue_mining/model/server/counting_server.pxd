from prolothar_queue_mining.model.server.server cimport Server
from prolothar_queue_mining.model.job cimport Job

cdef class CountingServer(Server):
    cdef int nr_of_served_jobs

    cpdef int get_nr_of_served_jobs(self)
