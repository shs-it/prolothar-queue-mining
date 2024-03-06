from math import log2

from prolothar_queue_mining.model.service_time.service_time import ServiceTime
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.job.regressor import Regressor
from prolothar_queue_mining.model.distribution import Distribution
from prolothar_queue_mining.model.distribution import C2dDistribution
from prolothar_queue_mining.model.distribution import ContinuousDistribution
from prolothar_queue_mining.model.distribution import DiscreteDistribution
from prolothar_queue_mining.model.distribution import DiscreteDegenerateDistribution

class ServiceTimeWithRegressor(ServiceTime):
    """
    a service time that uses a regression model and an error distribution
    to make predictions
    """

    def __init__(
            self, regressor: Regressor,
            error_distribution: Distribution):
        self.__regressor = regressor
        if isinstance(error_distribution, DiscreteDistribution):
            self.__error_distribution = error_distribution
        elif isinstance(error_distribution, ContinuousDistribution):
            self.__error_distribution = C2dDistribution(error_distribution)
        else:
            raise NotImplementedError(f'unsupported distribution type {error_distribution}')

    def get_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int:
        return max(0, self.__get_raw_prediction(job) +
            self.__error_distribution.get_next_sample()
        )

    def get_regressor(self) -> Regressor:
        return self.__regressor

    def get_batch_service_time(self, batch: list[Job], nr_of_jobs_in_system: int) -> int:
        return self.get_service_time(batch[0], nr_of_jobs_in_system)

    def copy(self) -> ServiceTime:
        return ServiceTimeWithRegressor(
            self.__regressor,
            self.__error_distribution)

    def copy_mean(self) -> 'ServiceTime':
        return ServiceTimeWithRegressor(
            self.__regressor,
            DiscreteDegenerateDistribution(round(self.__error_distribution.get_mean())))

    def compute_probability(self, x: int, job: Job, nr_of_jobs_in_system: int) -> float:
        return self.__error_distribution.compute_pmf(x - self.__get_raw_prediction(job))

    def compute_max_probability(self, x: int) -> float:
        #just assume that the regressor predicts x perfectly
        return self.__error_distribution.compute_pmf(self.__error_distribution.get_mode())

    def get_expected_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int:
        return max(0, self.__get_raw_prediction(job) + round(self.__error_distribution.get_mean()))

    def __get_raw_prediction(self, job: Job) -> int:
        return round(self.__regressor.predict(job))

    def get_most_likely_service_time(self, job: Job, nr_of_jobs_in_system: int) -> tuple[int, float]:
        return (
            max(0, self.__get_raw_prediction(job) + self.__error_distribution.get_mode()),
            self.__error_distribution.compute_pmf(self.__error_distribution.get_mode())
        )

    def set_seed(self, seed: int):
        self.__error_distribution.set_seed(seed)

    def get_mdl_of_model(self) -> float:
        mdl_of_model = self.__error_distribution.get_mdl_of_model()
        self.__regressor.get_mdl_of_model()
        return mdl_of_model

    def get_min_code_length_for_one_job(self) -> float:
        return -log2(self.__error_distribution.compute_pmf(self.__error_distribution.get_mode()))

    def is_deterministic(self) -> bool:
        return self.__error_distribution.is_deterministic()

    def __repr__(self):
        return f'ServiceTime({self.__regressor}, {self.__error_distribution})'
