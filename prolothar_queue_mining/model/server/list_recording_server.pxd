from prolothar_queue_mining.model.server.server cimport Server
from prolothar_queue_mining.model.job cimport Job

cdef class ListRecordingServer(Server):
    cdef list served_jobs

    cpdef list get_served_jobs(self)
