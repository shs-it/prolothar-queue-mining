from typing import List

from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.service_time import ServiceTime

cdef class CountingServer(Server):
    """
    extends Server for counting how many jobs have been served by this server
    """

    def __init__(self, service_time: ServiceTime):
        super(CountingServer, self).__init__(service_time)
        self.nr_of_served_jobs = 0

    cpdef int get_nr_of_served_jobs(self):
        return self.nr_of_served_jobs

    cpdef int get_service_time(self, Job job, int nr_of_jobs_in_system):
        self.nr_of_served_jobs += 1
        return super(CountingServer, self).get_service_time(job, nr_of_jobs_in_system)

    cpdef int get_batch_service_time(self, list batch, int nr_of_jobs_in_system):
        self.nr_of_served_jobs += len(batch)
        return super(CountingServer, self).get_batch_service_time(batch, nr_of_jobs_in_system)