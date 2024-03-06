from prolothar_queue_mining.model.job cimport Job

cdef class Server:
    cdef object service_time
    cdef Job current_job

    cpdef bint is_ready_for_service(self)
    cpdef set_current_job(self, Job job)
    cpdef set_seed(self, seed: int|None)
    cpdef Server copy(self)
    cpdef Server copy_mean(self)
    cpdef int get_service_time(self, Job job, int nr_of_jobs_in_system)
    cpdef int get_batch_service_time(self, list batch, int nr_of_jobs_in_system)

