from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.service_time.service_time import ServiceTime
from prolothar_queue_mining.model.job import Job

class OracleServiceTime(ServiceTime):
    """
    computes the service time from a already known exit time
    """

    def __init__(self, environment: Environment, exit_time_per_job: dict[Job, int]):
        """
        creates and intializes this OracleServiceTime instance

        Parameters
        ----------
        environment : Environment
            used to get the current time
        exit_time_per_job : dict[Job, float]
            used to get the exit time of a job. the service time is computed
            by the difference of exit time and current time
        """
        self.__environment = environment
        self.__exit_time_per_job = exit_time_per_job

    def get_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int:
        return max(0, self.__exit_time_per_job[job] - self.__environment.get_current_time())

    def get_expected_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int:
        return self.get_service_time(job)

    def get_most_likely_service_time(self, job: Job, nr_of_jobs_in_system: int) -> tuple[int, float]:
        return self.get_service_time(job), 1

    def get_batch_service_time(self, batch: list[Job], nr_of_jobs_in_system: int) -> float:
        return max(0, self.__exit_time_per_job[batch[-1]] - self.__environment.get_current_time())

    def compute_probability(self, x: int, job: Job, nr_of_jobs_in_system: int) -> float:
        return 1

    def compute_max_probability(self, x: int) -> float:
        return 1

    def copy(self) -> ServiceTime:
        return OracleServiceTime(self.__environment, dict(self.__exit_time_per_job))

    def set_seed(self, seed: int):
        #no randomness included
        pass

    def is_deterministic(self) -> bool:
        return True

    def __repr__(self):
        return f'OracleServiceTime({self.__exit_time_per_job})'
