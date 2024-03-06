from abc import ABC, abstractmethod

from statistics import median
from prolothar_common.experiments.statistics import Statistics

from prolothar_queue_mining.model.job import Job

class DepartureTimePredictor(ABC):
    """
    interface for departure time predictors
    """

    def predict(self, arrivals: list[tuple[Job, int]]) -> dict[Job, int]:
        """
        predicts the departure time for each job in the arrivals list
        """
        return self.predict_waiting_and_departure_times(arrivals)[1]

    def predict_waiting_and_departure_times(
            self, arrivals: list[tuple[Job, int]]) -> tuple[dict[Job, int]|None, dict[Job, int]]:
        """
        returns a 2-tuple. the first component is a dictionary with the predicted
        waiting times per job. this can be None if this predictor does not support
        prediction of waiting times. the second component is a dictionary with the
        predicted departure times.
        """
        waiting_times, exit_times = self.predict_waiting_and_departure_times_distribution(arrivals)
        return {
            job: Statistics(waiting_time_list).mean() for job, waiting_time_list
            in  waiting_times.items()
        } if waiting_times is not None else None, {
            job: Statistics(departure_time_list).mean() for job, departure_time_list
            in  exit_times.items()
        }

    @abstractmethod
    def predict_waiting_and_departure_times_distribution(
        self, arrivals: list[tuple[Job, int]]) -> tuple[dict[Job, list[int]]|None, dict[Job, list[int]]]:
        """
        predicts a list of possible waiting times and departure times for each job
        given a fixed list of arrival times.

        Returns
        -------
        tuple[dict[Job, list[int]], dict[Job, list[int]]]
            a list of predicted waiting times and a list of predicted departure times
            for each job
        """
