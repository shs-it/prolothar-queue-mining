from typing import List

from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.service_time import ServiceTime

cdef class Server:

    def __init__(self, service_time: ServiceTime):
        self.service_time = service_time
        self.current_job = None

    cpdef bint is_ready_for_service(self):
        return self.current_job is None

    cpdef set_current_job(self, Job job):
        self.current_job = job

    cpdef int get_service_time(self, Job job, int nr_of_jobs_in_system):
        try:
            return self.service_time.get_service_time(job, nr_of_jobs_in_system)
        except OverflowError as e:
            #e.g. gamma regressor tends to predict ridiculously large values
            # => set a limit
            # during model selection, our MDL score should filter out such pathologic models
            return 1_000_000

    cpdef int get_batch_service_time(self, list batch: List[Job], int nr_of_jobs_in_system):
        return self.service_time.get_batch_service_time(batch, nr_of_jobs_in_system)

    def get_service_time_name(self) -> str:
        """
        get a short name for the service time type of this server, which
        corresponds to "S" in Kendall's notation
        """
        return str(self.service_time)

    def get_service_time_definition(self) -> ServiceTime:
        return self.service_time

    cpdef Server copy(self):
        """
        returns a deep copy of this server with the same state.
        if an attribute of this server uses a random seed and the
        random seed is "None", this will result in different behavior during
        simulation. this is not a bug, but the reason for this method.
        otherwise use the "deepcopy" python module.
        """
        cdef Server copy = Server(self.service_time.copy())
        copy.set_current_job(self.current_job)
        return copy

    cpdef Server copy_mean(self):
        """
        returns a deep copy of this server with the same state.
        if an attribute of this server uses a random seed and the
        random seed is "None", this will result in different behavior during
        simulation. this is not a bug, but the reason for this method.
        otherwise use the "deepcopy" python module.
        if the underlying service time contains stochastic behavior, this will be
        removed by setting it to the mean.
        """
        cdef Server copy = Server(self.service_time.copy_mean())
        copy.set_current_job(self.current_job)
        return copy

    cpdef set_seed(self, seed: int|None):
        """
        sets the seed of the service time
        """
        self.service_time.set_seed(seed)
