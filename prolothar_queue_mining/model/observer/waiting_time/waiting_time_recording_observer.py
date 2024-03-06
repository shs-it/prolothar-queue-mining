import numpy as np
from prolothar_common.experiments.statistics import Statistics

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.observer.waiting_time import WaitingTimeObserver

class WaitingTimeRecordingObserver(WaitingTimeObserver):
    """
    an observer for (statistical) analysis of recorded waiting times over time
    """

    def __init__(self, min_service_time: int = 0):
        self.__statistics = Statistics()
        self.__raw_waiting_times: list[int] = []
        self.__arrival_times: list[int] = []
        self.__min_service_time = min_service_time
        self.__waiting_time_per_job: dict[Job, int] = {}

    def notify(self, job: Job, arrival_time: int, start_of_service_time: int):
        if start_of_service_time >= self.__min_service_time:
            waiting_time = start_of_service_time - arrival_time
            self.__statistics.push(waiting_time)
            self.__raw_waiting_times.append(waiting_time)
            self.__arrival_times.append(arrival_time)
            self.__waiting_time_per_job[job] = waiting_time

    def get_waiting_time_per_job_dict(self) -> dict[Job, int]:
        return self.__waiting_time_per_job

    def get_max_waiting_time(self) -> int:
        """
        returns the maximal observed waiting time
        """
        return self.__statistics.maximum()

    def get_mean_waiting_time(self) -> float:
        """
        returns the mean observed waiting time
        """
        return self.__statistics.mean()

    def get_median_waiting_time(self) -> float:
        """
        returns the median observed waiting time. if no value is available,
        np.nan is returned
        """
        if self.__raw_waiting_times:
            return np.median(self.__raw_waiting_times)
        else:
            return np.nan

    def get_timeseries_data(self) -> tuple[list[int], list[int]]:
        """
        returns a list of arrival times and corresponding waiting times.
        the list of arrival times can contain duplicates if two or more jobs arrived
        at the same time.
        """
        return self.__arrival_times, self.__raw_waiting_times
