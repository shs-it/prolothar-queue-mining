from prolothar_common.experiments.statistics import Statistics

from prolothar_queue_mining.model.population import ListPopulation
from prolothar_queue_mining.model.arrival_process.arrival_process import ArrivalProcess
from prolothar_queue_mining.model.arrival_process.fixed_arrival import FixedArrival
from prolothar_queue_mining.model.job import Job

class RecordingArrival(ArrivalProcess):
    """
    a meta arrival process, where the arrival times of jobs is recorded for
    later reuse
    """

    def __init__(self, arrival: ArrivalProcess):
        self.__arrival = arrival
        self.__recorded_arrival_times: list[int] = []
        self.__recorded_jobs: list[Job] = []
        self.__interarrival_statistics = Statistics()

    def get_next_job(self) -> tuple[int,Job]:
        arrival_time, job = self.__arrival.get_next_job()
        self._add_recording(job, arrival_time)
        return arrival_time, job

    def _add_recording(self, job: Job, arrival_time: int):
        if self.__recorded_arrival_times:
            self.__interarrival_statistics.push(arrival_time - self.__recorded_arrival_times[-1])
        self.__recorded_arrival_times.append(arrival_time)
        self.__recorded_jobs.append(job)

    def get_mean_arrival_rate(self) -> float:
        return self.__arrival.get_mean_arrival_rate()

    def get_arrival_process(self) -> ArrivalProcess:
        return self.__arrival

    def get_recorded_arrival_times(self) -> list[int]:
        """
        returns a list of recorded arrival times
        """
        return self.__recorded_arrival_times

    def get_recorded_jobs(self) -> list[Job]:
        """
        returns a list of recorded jobs in the order of their recorded arrival times
        """
        return self.__recorded_jobs

    def get_min_interarrival_time(self) -> int:
        return self.__interarrival_statistics.minimum()

    def get_max_interarrival_time(self) -> int:
        return self.__interarrival_statistics.maximum()

    def get_mean_interarrival_time(self) -> float:
        return self.__interarrival_statistics.mean()

    def get_stddev_interarrival_time(self) -> float:
        return self.__interarrival_statistics.stddev()

    def to_fixed_arrival_process(self) -> FixedArrival:
        return FixedArrival(
            ListPopulation(list(self.__recorded_jobs)),
            list(self.__recorded_arrival_times))

    def copy(self) -> ArrivalProcess:
        copy = RecordingArrival(self.__arrival.copy())
        for arrival_time, job in zip(self.__recorded_arrival_times, self.__recorded_jobs):
            copy._add_recording(job, arrival_time)
        return copy

    def set_seed(self, seed: int):
        self.__arrival.set_seed(seed)

    def __repr__(self):
        return f'RecordingArrival({self.__arrival})'
