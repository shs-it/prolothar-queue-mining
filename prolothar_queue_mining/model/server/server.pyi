from prolothar_queue_mining.model.job import Job

from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.service_time import ServiceTime

class Server:
    current_job: Job|None

    def __init__(self, service_time: ServiceTime): ...

    def is_ready_for_service(self) -> bool: ...

    def set_current_job(self, job: Job): ...

    def get_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int: ...

    def get_batch_service_time(self, batch: list[Job], nr_of_jobs_in_system: int) -> int: ...

    def get_service_time_name(self) -> str:
        """
        get a short name for the service time type of this server, which
        corresponds to "S" in Kendall's notation
        """
        ...

    def get_service_time_definition(self) -> ServiceTime: ...

    def copy(self) -> Server:
        """
        returns a deep copy of this server with the same state.
        if an attribute of this server uses a random seed and the
        random seed is "None", this will result in different behavior during
        simulation. this is not a bug, but the reason for this method.
        otherwise use the "deepcopy" python module.
        """
        ...

    def copy_mean(self) -> Server:
        """
        returns a deep copy of this server with the same state.
        if an attribute of this server uses a random seed and the
        random seed is "None", this will result in different behavior during
        simulation. this is not a bug, but the reason for this method.
        otherwise use the "deepcopy" python module.
        if the underlying service time contains stochastic behavior, this will be
        removed by setting it to the mean.
        """
        ...

    def set_seed(self, seed: int|None):
        """
        sets the seed of the service time
        """
        ...
