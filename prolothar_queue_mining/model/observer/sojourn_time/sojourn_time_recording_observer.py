import numpy as np
from prolothar_common.experiments.statistics import Statistics

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.observer.waiting_time import WaitingTimeObserver

class SojournTimeRecordingObserver(WaitingTimeObserver):
    """
    an observer for (statistical) analysis of recorded sojourn times over time
    """

    def __init__(self, min_exit_time: int = 0):
        self.__statistics = Statistics()
        self.__raw_sojourn_times: list[int] = []
        self.__arrival_times: list[int] = []
        self.__min_exit_time = min_exit_time
        self.__sojourn_time_per_job: dict[Job, int] = {}

    def notify(self, job: Job, arrival_time: int, exit_time: int):
        if exit_time >= self.__min_exit_time:
            sojourn_time = exit_time - arrival_time
            self.__statistics.push(sojourn_time)
            self.__raw_sojourn_times.append(sojourn_time)
            self.__arrival_times.append(arrival_time)
            self.__sojourn_time_per_job[job] = sojourn_time

    def get_sojourn_time_per_job_dict(self) -> dict[Job, int]:
        return self.__sojourn_time_per_job

    def get_sojourn_times(self) -> list[int]:
        return self.__raw_sojourn_times

    def get_max_sojourn_time(self) -> int:
        """
        returns the maximal observed sojourn time
        """
        return self.__statistics.maximum()

    def get_min_sojourn_time(self) -> int:
        """
        returns the minimal observed sojourn time
        """
        return self.__statistics.minimum()

    def get_mean_sojourn_time(self) -> float:
        """
        returns the mean observed sojourn time
        """
        return self.__statistics.mean()

    def get_stddev_sojourn_time(self) -> float:
        """
        returns the standard deviation of the observed sojourn time
        """
        return self.__statistics.stddev()

    def get_median_sojourn_time(self) -> float:
        """
        returns the median observed sojourn time. if no value is available,
        np.nan is returned
        """
        if self.__raw_sojourn_times:
            return np.median(self.__raw_sojourn_times)
        else:
            return np.nan

    def get_timeseries_data(self) -> tuple[list[int], list[int]]:
        """
        returns a list of arrival times and corresponding sojourn times.
        the list of arrival times can contain duplicates if two or more jobs arrived
        at the same time.
        """
        return self.__arrival_times, self.__raw_sojourn_times
