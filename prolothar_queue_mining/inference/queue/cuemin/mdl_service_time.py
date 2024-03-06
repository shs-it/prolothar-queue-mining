from math import log2
import numpy as np

from prolothar_common.mdl_utils import L_N, prequential_coding_length

from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.service_time.service_time import ServiceTime
from prolothar_queue_mining.model.distribution.distribution import Distribution
from prolothar_queue_mining.model.job import Job

class MdlServiceTime(ServiceTime):
    """
    computes the service time from a already known exit time and computes
    a fit of data score from an underlying distribution
    """

    def __init__(self, environment: Environment, service_time: ServiceTime, exit_time_per_job: dict[Job, int]):
        """
        creates and intializes this OracleServiceTime instanz

        Parameters
        ----------
        environment : Environment
            used to get the current time
        service_time : ServiceTime
            used to compute a fitness score
        exit_time_per_job : dict[Job, float]
            used to get the exit time of a job. the service time is computed
            by the difference of exit time and current time
        """
        self.__environment = environment
        self.__service_time = service_time
        self.__exit_time_per_job = exit_time_per_job
        self.__nr_of_jobs_with_departure_time = 0
        self.__nr_of_jobs_with_unknown_departure_time = 0
        self.__total_length_of_predicted_value_codes: float = 0
        self.__nr_of_positive_residuals = 0
        self.__nr_of_negative_residuals = 0
        self.__nr_of_zero_residuals = 0
        self.__total_length_of_residual_codes = 0

    def get_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int:
        try:
            required_service_time = self.__exit_time_per_job[job] - self.__environment.get_current_time()
            self.__nr_of_jobs_with_departure_time += 1
        except KeyError:
            self.__nr_of_jobs_with_unknown_departure_time += 1
            return self.__service_time.get_most_likely_service_time(
                job, nr_of_jobs_in_system)[0]
        if required_service_time < 0:
            predicted_service_time = self.__service_time.get_most_likely_service_time(
                job, nr_of_jobs_in_system)[0]
            residual = required_service_time
        else:
            probability = self.__service_time.compute_probability(
                required_service_time, job, nr_of_jobs_in_system)
            if probability <= Distribution.ALLMOST_ZERO:
                predicted_service_time, probability = self.__service_time.get_most_likely_service_time(
                    job, nr_of_jobs_in_system)
                residual = required_service_time - predicted_service_time
            else:
                predicted_service_time = required_service_time
                residual = 0
            self.__total_length_of_predicted_value_codes -= log2(probability)
        self.__encode_residual(residual)
        return predicted_service_time

    def get_batch_service_time(self, batch: list[Job], nr_of_jobs_in_system: int) -> int:
        try:
            required_service_time = self.__exit_time_per_job[batch[0]] - self.__environment.get_current_time()
        except KeyError:
            self.__nr_of_jobs_with_unknown_departure_time += len(batch)
            return self.__service_time.get_most_likely_service_time(batch[0], nr_of_jobs_in_system)[0]

        if required_service_time < 0:
            predicted_service_time = 0
        else:
            probability = self.__service_time.compute_probability(
                required_service_time, batch[0], nr_of_jobs_in_system)
            if probability <= Distribution.ALLMOST_ZERO:
                predicted_service_time, probability = self.__service_time.get_most_likely_service_time(
                    batch[0], nr_of_jobs_in_system)
            else:
                predicted_service_time = required_service_time
            self.__total_length_of_predicted_value_codes -= log2(probability)
        predicted_departure_time = self.__environment.get_current_time() + predicted_service_time
        for job in batch:
            try:
                residual = self.__exit_time_per_job[job] - predicted_departure_time
                self.__encode_residual(residual)
                self.__nr_of_jobs_with_departure_time += 1
            except KeyError:
                self.__nr_of_jobs_with_unknown_departure_time += 1
        return predicted_service_time

    def __encode_residual(self, residual: int):
        residual = int(residual)
        residual_sign = np.sign(residual)
        if residual_sign == 0:
            self.__nr_of_zero_residuals += 1
        elif residual_sign == -1:
            self.__nr_of_negative_residuals += 1
            self.__total_length_of_residual_codes += L_N(-residual)
        else:
            self.__nr_of_positive_residuals += 1
            self.__total_length_of_residual_codes += L_N(residual)

    def get_total_encoded_length(self) -> float:
        # return (
        #     prequential_coding_length({
        #         0: self.__nr_of_jobs_with_departure_time,
        #         1: self.__nr_of_jobs_with_unknown_departure_time
        #     }) + self.__total_length_of_predicted_value_codes +
        #     prequential_coding_length({
        #         -1: self.__nr_of_negative_residuals,
        #         0: self.__nr_of_zero_residuals,
        #         1: self.__nr_of_positive_residuals
        #     }) + self.__total_length_of_residual_codes
        # )
        # return self.__total_length_of_predicted_value_codes + self.__total_length_of_residual_codes
        return (
            self.__total_length_of_predicted_value_codes +
            prequential_coding_length({
                -1: self.__nr_of_negative_residuals,
                0: self.__nr_of_zero_residuals,
                1: self.__nr_of_positive_residuals
            }) + self.__total_length_of_residual_codes
        )

    def get_total_length_of_residual_codes(self) -> float:
        return self.__total_length_of_residual_codes

    def get_total_length_of_value_codes(self) -> float:
        return self.__total_length_of_predicted_value_codes

    def copy(self) -> ServiceTime:
        return MdlServiceTime(self.__environment, self.__service_time.copy(), dict(self.__exit_time_per_job))

    def compute_probability(self, x: int, job: Job, nr_of_jobs_in_system: int) -> float:
        raise NotImplementedError('should never be used outside model search')

    def compute_max_probability(self, x: int) -> float:
        raise NotImplementedError('should never be used outside model search')

    def get_expected_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int:
        raise NotImplementedError('should never be used outside model search')

    def get_most_likely_service_time(self, job: Job, nr_of_jobs_in_system: int) -> tuple[int, float]:
        raise NotImplementedError('should never be used outside model search')

    def is_deterministic(self) -> bool:
        return True

    def set_seed(self, seed: int):
        self.__service_time.set_seed(seed)

    def get_service_time_model(self) -> ServiceTime:
        return self.__service_time

    def __repr__(self):
        return f'MdlServiceTime({self.__service_time})'
