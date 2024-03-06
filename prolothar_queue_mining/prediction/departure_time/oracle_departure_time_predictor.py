from prolothar_queue_mining.prediction.departure_time.departure_time_predictor import DepartureTimePredictor

from prolothar_queue_mining.model.job import Job

class OracleDepartureTimePredictor(DepartureTimePredictor):
    """
    dummy predictor that knows the true departure time for each job
    """
    def __init__(self, departure_time_per_job: dict[Job, int]):
        self.__departure_time_per_job = departure_time_per_job

    def predict_waiting_and_departure_times_distribution(
            self, arrivals: list[tuple[Job, int]]) -> tuple[dict[Job, list[int]]|None, dict[Job, list[int]]]:
        return None, {
            job: [self.__departure_time_per_job.get(job, None)] for job,_ in arrivals
        }