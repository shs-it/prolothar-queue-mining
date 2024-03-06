from collections import defaultdict
import numpy as np
from sklearn.base import ClusterMixin
from tqdm import tqdm
from methodtools import lru_cache

from prolothar_queue_mining.prediction.departure_time.departure_time_predictor import DepartureTimePredictor

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.service_time import ServiceTime
from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.event import Event


class LoadClusterDepartureTimePredictor(DepartureTimePredictor):
    """
    a predictor that predicts sojourn time based on the number of jobs in the
    system
    """

    def __init__(
            self, cluster_predictor: ClusterMixin, predictor_per_cluster: dict[int, ServiceTime],
            nr_of_repetitions: int = 1000, show_progress_bar: bool = False):
        self.__cluster_predictor = cluster_predictor
        self.__predictor_per_cluster = predictor_per_cluster
        self.__nr_of_repetitions = nr_of_repetitions
        self.__show_progress_bar = show_progress_bar

    def set_nr_of_repetitions(self, nr_of_repetitions: int):
        self.__nr_of_repetitions = nr_of_repetitions

    def set_show_progress_bar(self, show_progress_bar: bool):
        self.__show_progress_bar = show_progress_bar

    def predict_waiting_and_departure_times_distribution(
            self, arrivals: list[tuple[Job, int]]) -> tuple[dict[Job, list[int]] | None, dict[Job, list[int]]]:
        exit_times_per_job = defaultdict(list)

        repetition_range = range(self.__nr_of_repetitions)
        if self.__show_progress_bar:
            repetition_range = tqdm(repetition_range)

        for _ in repetition_range:
            jobs_in_system = set()

            environment = Environment()
            for job, arrival_time in arrivals:
                environment.schedule_event(ArrivalEvent(
                    job, arrival_time, jobs_in_system, exit_times_per_job, self))
            environment.run_until_event_queue_is_empty()

        return None, exit_times_per_job

    def predict_sojourn_time_of_job(self, job: Job, nr_of_jobs_in_system: int) -> int:
        cluster_label = self.__predict_cluster_label(nr_of_jobs_in_system)
        return self.__predictor_per_cluster[cluster_label].get_service_time(job, nr_of_jobs_in_system)

    @lru_cache()
    def __predict_cluster_label(self, nr_of_jobs_in_system: int) -> int:
        return self.__cluster_predictor.predict_single(nr_of_jobs_in_system)

class ArrivalEvent(Event):

    def __init__(
            self, job: Job, arrival_time: int, jobs_in_system: set[Job],
            exit_times_per_job: dict[Job, list], parent: LoadClusterDepartureTimePredictor):
        super().__init__(arrival_time, prio=1)
        self.__job = job
        self.__jobs_in_system = jobs_in_system
        self.__parent = parent
        self.__exit_times_per_job = exit_times_per_job

    def execute(self, environment: Environment):
        exit_time = self.time + self.__parent.predict_sojourn_time_of_job(
                self.__job, len(self.__jobs_in_system))
        environment.schedule_event(DepartureEvent(
            self.__job, exit_time, self.__jobs_in_system
        ))
        self.__exit_times_per_job[self.__job].append(exit_time)
        self.__jobs_in_system.add(self.__job)

class DepartureEvent(Event):

    def __init__(
            self, job: Job, departure_time: int, jobs_in_system: set[Job]):
        super().__init__(departure_time)
        self.__job = job
        self.__jobs_in_system = jobs_in_system

    def execute(self, environment: Environment):
        self.__jobs_in_system.remove(self.__job)