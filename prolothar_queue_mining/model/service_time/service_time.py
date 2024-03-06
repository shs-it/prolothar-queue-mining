from abc import ABC, abstractmethod
from prolothar_queue_mining.model.job import Job

class ServiceTime(ABC):
    """
    distribution or computation of the service time of jobs
    """

    @abstractmethod
    def get_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int:
        """
        determines the service (processing) time of the given job
        """

    @abstractmethod
    def get_expected_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int:
        """
        determines the expected value of the service (processing) time of the given job
        """

    @abstractmethod
    def get_most_likely_service_time(self, job: Job, nr_of_jobs_in_system: int) -> tuple[int, float]:
        """
        determines the most likely (mode) value of the service (processing) time of the given job.
        returns the service time together with the probability
        """

    @abstractmethod
    def get_batch_service_time(self, batch: list[Job], nr_of_jobs_in_system: int) -> int:
        """
        determines the service (processing) time of the given batch of jobs
        """

    @abstractmethod
    def compute_probability(self, x: int, job: Job, nr_of_jobs_in_system: int) -> float:
        """
        computes the probability of an observated service time for a given job
        """

    @abstractmethod
    def compute_max_probability(self, x: int) -> float:
        """
        computes the maximal probability of an observated service time
        for all possible jobs and number of jobs in the system
        """

    @abstractmethod
    def set_seed(self, seed: int):
        """
        sets the seed for any random number generator used by this service time object
        """

    @abstractmethod
    def is_deterministic(self) -> bool:
        """
        returns True iff calling "get_service_time" for the same job always
        leads to the same result
        """

    def get_mdl_of_model(self) -> float:
        """
        returns the encoded length of the model if supported. otherweise raises a
        NotImplementedError
        """
        raise NotImplementedError()

    def get_min_code_length_for_one_job(self) -> float:
        """
        returns the minimal code length to encode the service time of a job.
        this can be used to compute a lower bound for L(D|M).
        if not supported, this methoded raises a NotImplementedError
        """
        raise NotImplementedError()

    @abstractmethod
    def copy(self) -> 'ServiceTime':
        """
        returns a deep copy of this service time with the same state.
        if an attribute of this service time uses a random seed and the
        random seed is "None", this will result in different behavior during
        simulation. this is not a bug, but the reason for this method.
        otherwise use the "deepcopy" python module.
        """

    def copy_mean(self) -> 'ServiceTime':
        """
        returns a deep copy of this service time with the same state.
        if an attribute of this service time uses a random seed and the
        random seed is "None", this will result in different behavior during
        simulation. this is not a bug, but the reason for this method.
        otherwise use the "deepcopy" python module.
         if this service time contains stochastic behavior, this will be
        removed by setting it to the mean.
        """
        if not self.is_deterministic():
            raise NotImplementedError()
        else:
            return self.copy()