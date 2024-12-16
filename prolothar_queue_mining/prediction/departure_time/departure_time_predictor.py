from abc import ABC, abstractmethod

from statistics import median, mode
from typing import Literal
from prolothar_common.experiments.statistics import Statistics

from prolothar_queue_mining.model.job import Job

LocationParameter = Literal['mean', 'mode', 'median']

class DepartureTimePredictor(ABC):
    """
    interface for departure time predictors
    """

    def predict(self, arrivals: list[tuple[Job, int]], location_parameter: LocationParameter = 'mean') -> dict[Job, int]:
        """
        predicts the departure time for each job in the arrivals list.

        the optional location_parameter determines the statistical function to compute a point prediction from an empirical distribution.
        """
        return self.predict_waiting_and_departure_times(arrivals, location_parameter=location_parameter)[1]

    def predict_waiting_and_departure_times(
            self, arrivals: list[tuple[Job, int]],
            location_parameter: LocationParameter = 'mean') -> tuple[dict[Job, int]|None, dict[Job, int]]:
        """
        returns a 2-tuple. the first component is a dictionary with the predicted
        waiting times per job. this can be None if this predictor does not support
        prediction of waiting times. the second component is a dictionary with the
        predicted departure times.
        """
        waiting_times, exit_times = self.predict_waiting_and_departure_times_distribution(arrivals)
        if location_parameter == 'mean':
            return {
                job: Statistics(waiting_time_list).mean() for job, waiting_time_list
                in  waiting_times.items()
            } if waiting_times is not None else None, {
                job: Statistics(departure_time_list).mean() for job, departure_time_list
                in  exit_times.items()
            }
        elif location_parameter == 'mode':
            return {
                job: mode(waiting_time_list) for job, waiting_time_list
                in  waiting_times.items()
            } if waiting_times is not None else None, {
                job: mode(departure_time_list) for job, departure_time_list
                in  exit_times.items()
            }            
        elif location_parameter == 'median':
            return {
                job: median(waiting_time_list) for job, waiting_time_list
                in  waiting_times.items()
            } if waiting_times is not None else None, {
                job: median(departure_time_list) for job, departure_time_list
                in  exit_times.items()
            }
        else:
            raise NotImplementedError(f'unknown location parameter: {location_parameter}')       

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
