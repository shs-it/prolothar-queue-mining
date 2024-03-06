from prolothar_queue_mining.prediction.departure_time.departure_time_predictor import DepartureTimePredictor

from prolothar_queue_mining.model.job import Job

class FixedSojournTimeDepartureTimePredictor(DepartureTimePredictor):
    """
    dummy predictor that predicts departure time for each job by adding
    a constant sojourn time to the arrival time
    """
    def __init__(self, sojourn_time: int):
        self.__sojourn_time = sojourn_time

    def predict_waiting_and_departure_times_distribution(
            self, arrivals: list[tuple[Job, int]]) -> tuple[dict[Job, list[int]] | None, dict[Job, list[int]]]:
        return None, {
            job: [arrival_time + self.__sojourn_time] for job,arrival_time in arrivals
        }
