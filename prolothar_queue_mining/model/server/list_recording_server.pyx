from typing import List

from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.service_time import ServiceTime

cdef class ListRecordingServer(Server):
    """
    extends Server for recording the order of served jobs
    """

    def __init__(self, service_time: ServiceTime):
        super(ListRecordingServer, self).__init__(service_time)
        self.served_jobs = []

    cpdef int get_service_time(self, Job job, int nr_of_jobs_in_system):
        self.served_jobs.append(job)
        return super(ListRecordingServer, self).get_service_time(job, nr_of_jobs_in_system)

    cpdef int get_batch_service_time(self, list batch, int nr_of_jobs_in_system):
        self.served_jobs.append(batch)
        return super(ListRecordingServer, self).get_batch_service_time(batch, nr_of_jobs_in_system)

    cpdef list get_served_jobs(self):
        return self.served_jobs
