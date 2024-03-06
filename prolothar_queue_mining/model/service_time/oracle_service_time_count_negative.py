from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.service_time.service_time import ServiceTime
from prolothar_queue_mining.model.job import Job

class OracleServiceTimeCountNegative(ServiceTime):
    """
    computes the service time from a already known exit time. counts how
    often the service time would have been < 0, which is a hint for not enough
    servers in the queue.
    """

    def __init__(self, environment: Environment, exit_time_per_job: dict[Job, int]):
        """
        creates and intializes this OracleServiceTimeCountNegative instance

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
        self.__nr_of_jobs_with_negative_service_times_in_a_row = 0
        self.__max_nr_of_jobs_with_negative_service_times_in_a_row = 0

    def get_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int:
        service_time = self.__exit_time_per_job[job] - self.__environment.get_current_time()
        if service_time < 0:
            self.__nr_of_jobs_with_negative_service_times_in_a_row += 1
            if self.__nr_of_jobs_with_negative_service_times_in_a_row > \
            self.__max_nr_of_jobs_with_negative_service_times_in_a_row:
                self.__max_nr_of_jobs_with_negative_service_times_in_a_row = \
                    self.__nr_of_jobs_with_negative_service_times_in_a_row
            return 0
        else:
            self.__nr_of_jobs_with_negative_service_times_in_a_row = 0
            return service_time

    def get_max_nr_of_jobs_with_negative_service_times(self) -> int:
        return self.__max_nr_of_jobs_with_negative_service_times_in_a_row

    def get_expected_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int:
        return self.get_service_time(job)

    def get_most_likely_service_time(self, job: Job, nr_of_jobs_in_system: int) -> tuple[int, float]:
        return self.get_service_time(job), 1

    def get_batch_service_time(self, batch: list[Job], nr_of_jobs_in_system: int) -> float:
        raise NotImplementedError('batch service not supported')

    def compute_probability(self, x: int, job: Job, nr_of_jobs_in_system: int) -> float:
        return 1

    def compute_max_probability(self, x: int) -> float:
        return 1

    def copy(self) -> ServiceTime:
        return OracleServiceTimeCountNegative(self.__environment, dict(self.__exit_time_per_job))

    def set_seed(self, seed: int):
        #no randomness included
        pass

    def is_deterministic(self) -> bool:
        return True

    def __repr__(self):
        return f'OracleServiceTimeCountNegative({self.__exit_time_per_job})'
