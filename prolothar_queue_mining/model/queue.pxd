from prolothar_queue_mining.model.environment cimport Environment
from prolothar_queue_mining.model.server cimport Server
from prolothar_queue_mining.model.job cimport Job

cdef class Queue:
    cdef __arrival_process
    cdef __servers
    cdef __exit
    cdef __waiting_area
    cdef __batch_size_distribution
    cdef __waiting_time_observer
    cdef __sojourn_time_observer
    cdef __queue_length_observer
    cdef __arrival_time_of_open_jobs
    cdef int __current_required_batch_size
    cdef bint __is_batch_service_possible
    cdef int __nr_of_jobs_in_system

    cpdef handle_job_arrival(self, Environment environment, int arrival_time, Job job)
    cdef try_to_serve_next_job(self, Environment environment)
    cdef __serve_job(self, Job job, Server server, int exit_time, Environment environment)
    cdef __serve_batch(self, list batch, Server server, int exit_time, Environment environment)
    cdef handle_batch_exit(self, Environment environment, Server server, int exit_time, list batch)
    cpdef handle_job_exit(self, Environment environment, Server server, int exit_time, Job job)
    cpdef schedule_next_arrival(self, Environment environment)