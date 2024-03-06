from abc import abstractmethod
from typing import Iterable

from sklearn.base import ClassifierMixin
from sklearn.model_selection import GridSearchCV

from prolothar_queue_mining.inference.queue.waiting_area.waiting_area_estimator import WaitingAreaEstimator
from prolothar_queue_mining.model.job.job_to_vector_transformer import JobToVectorTransformer

from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.waiting_area import PairwisePriorityClassifierWaitingArea
from prolothar_queue_mining.model.waiting_area.pairwise_priority_classifier import SklearnPairwisePriorityClassifier
from prolothar_queue_mining.model.event import Event
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.job.job_to_vector_transformer import MinMaxScaler
from prolothar_queue_mining.model.environment import Environment

class PairwiseSklearnCvEstimator(WaitingAreaEstimator):

    def __init__(
            self, numerical_feature_names: str, categorical_feature_names: str,
            classifier: ClassifierMixin, parameter_grid: dict[str, list]):
        self.__categorical_feature_names = categorical_feature_names
        self.__numerical_feature_names = numerical_feature_names
        self.__classifier = classifier
        self.__parameter_grid = parameter_grid

    def infer_waiting_area(
            self, observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]]) -> WaitingArea:

        departure_time_per_job = dict(observed_departures)
        max_sojourn_time = max(
            departure_time_per_job[job] - arrival_time
            for job, arrival_time in observed_arrivals
            if job in departure_time_per_job
        )
        scaler_for_arrival_time_difference = MinMaxScaler(
            min_value=-max_sojourn_time, max_value=max_sojourn_time)

        job_to_vector_transformer = self._create_job_to_vector_transformer(
            self.__numerical_feature_names, self.__categorical_feature_names,
            departure_time_per_job.keys())

        grid_search_cv = GridSearchCV(self.__classifier, self.__parameter_grid, cv=5)

        waiting_area = PairwisePriorityClassifierWaitingArea(
            SklearnPairwisePriorityClassifier(
                grid_search_cv,
                job_to_vector_transformer,
                scaler_for_arrival_time_difference=scaler_for_arrival_time_difference
            )
        )

        environment = Environment()
        for job, arrival_time in observed_arrivals:
            environment.schedule_event(ArrivalEvent(job, arrival_time, waiting_area))
        for job, departure_time in observed_departures:
            environment.schedule_event(DepartureEvent(job, departure_time, waiting_area))
        environment.run_timesteps(observed_departures[-1][1])

        waiting_area.learn_classifier()
        return PairwisePriorityClassifierWaitingArea(
            SklearnPairwisePriorityClassifier(
                grid_search_cv.best_estimator_,
                job_to_vector_transformer,
                scaler_for_arrival_time_difference=scaler_for_arrival_time_difference
            )
        )

    @abstractmethod
    def _create_job_to_vector_transformer(
            self, numerical_feature_names: list[str], categorical_feature_names: list[str],
            jobs: Iterable[Job]):
        pass

class ArrivalEvent(Event):

    def __init__(self, job: Job, arrival_time: int, waiting_area: PairwisePriorityClassifierWaitingArea):
        super().__init__(arrival_time)
        self.__job = job
        self.__waiting_area = waiting_area

    def execute(self, environment: Environment):
        self.__waiting_area.add_job_for_learning(self.time, self.__job)

class DepartureEvent(Event):

    def __init__(self, job: Job, departure_time: int, waiting_area: PairwisePriorityClassifierWaitingArea):
        super().__init__(departure_time)
        self.__job = job
        self.__waiting_area = waiting_area

    def execute(self, environment: Environment):
        self.__waiting_area.pop_next_job_for_learning(self.__job.job_id)
