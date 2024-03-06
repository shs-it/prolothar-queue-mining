from methodtools import lru_cache

from prolothar_common.mdl_utils import L_N

from prolothar_queue_mining.model.service_time.service_time import ServiceTime
from prolothar_queue_mining.model.job import Job

class LoadDependentServiceTime(ServiceTime):
    """
    depending on the system-load, i.e. the number of jobs in the system, a
    different sub service time model is used
    """

    def __init__(
        self,
        sub_service_time_list: list[ServiceTime],
        load_threshold_list: list[int]):
        """
        creates and intializes this LoadDependentServiceTime instance

        Parameters
        ----------
        sub_service_time_list: list[ServiceTime]
            service times used for increasing system load
        load_threshold_list: list[int]
            thresholds that determine which sub service time model should be used
            depending on the number of jobs in the system.
            [1,3] means there are 3 sub models:
            - the first sub model is used if the system is empty,
            - the second sub model is used if at most 3 jobs are in the system
            - and the third sub model is used if there are more than 3 jobs in the system
        """
        if len(sub_service_time_list) < 2:
            raise ValueError('sub_service_time_list must not contain less than 2 elements')
        if len(sub_service_time_list) != len(load_threshold_list) + 1:
            raise ValueError(
                'sub_service_time_list must contain exactly 1 element more than load_threshold_list')
        if load_threshold_list[0] <= 0:
            raise ValueError(f'load_threshold_list must not contain entries <= 0: {load_threshold_list}')
        if load_threshold_list != sorted(load_threshold_list):
            raise ValueError('load_threshold_list must be sorted')
        self.__sub_service_time_list = sub_service_time_list
        self.__load_threshold_list = load_threshold_list

    @lru_cache()
    def __get_current_sub_model(self, nr_of_jobs_in_system: int) -> ServiceTime:
        for load_threshold, service_time in zip(self.__load_threshold_list, self.__sub_service_time_list):
            if nr_of_jobs_in_system <= load_threshold:
                return service_time
        return self.__sub_service_time_list[-1]

    def get_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int:
        return self.__get_current_sub_model(
            nr_of_jobs_in_system
        ).get_service_time(job, nr_of_jobs_in_system)

    def get_expected_service_time(self, job: Job, nr_of_jobs_in_system: int) -> int:
        return self.__get_current_sub_model(
            nr_of_jobs_in_system
        ).get_expected_service_time(job, nr_of_jobs_in_system)

    def get_most_likely_service_time(self, job: Job, nr_of_jobs_in_system: int) -> tuple[int, float]:
        return self.__get_current_sub_model(
            nr_of_jobs_in_system
        ).get_most_likely_service_time(job, nr_of_jobs_in_system)

    def get_batch_service_time(self, batch: list[Job], nr_of_jobs_in_system: int) -> float:
        return self.__get_current_sub_model(
            nr_of_jobs_in_system
        ).get_batch_service_time(batch, nr_of_jobs_in_system)

    def compute_probability(self, x: int, job: Job, nr_of_jobs_in_system: int) -> float:
        return self.__get_current_sub_model(
            nr_of_jobs_in_system
        ).compute_probability(x, job, nr_of_jobs_in_system)

    @lru_cache()
    def compute_max_probability(self, x: int) -> float:
        return max(s.compute_max_probability(x) for s in  self.__sub_service_time_list)

    def copy(self) -> ServiceTime:
        #for effciency reasons (we want to have maximale usage of caches)
        return self

    def copy_mean(self) -> 'ServiceTime':
        return LoadDependentServiceTime(
            [s.copy_mean() for s in self.__sub_service_time_list],
            self.__load_threshold_list
        )

    def get_mdl_of_model(self) -> float:
        mdl_of_model = L_N(len(self.__sub_service_time_list))
        for threshold in self.__load_threshold_list:
            mdl_of_model += L_N(int(threshold))
        for submodel in self.__sub_service_time_list:
            mdl_of_model += submodel.get_mdl_of_model()
        return mdl_of_model

    def get_min_code_length_for_one_job(self) -> float:
        return min(
            submodel.get_min_code_length_for_one_job()
            for submodel in self.__sub_service_time_list
        )

    def set_seed(self, seed: int):
        for submodel in self.__sub_service_time_list:
            submodel.set_seed(seed)

    def is_deterministic(self) -> bool:
        return all(submodel.is_deterministic() for submodel in self.__sub_service_time_list)

    def get_sub_service_time_list(self) -> list[ServiceTime]:
        return self.__sub_service_time_list

    def __repr__(self):
        return f'LoadDependentServiceTime({self.__sub_service_time_list}, {self.__load_threshold_list})'
