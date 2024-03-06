import numpy as np
from sklearn.base import RegressorMixin

from prolothar_queue_mining.prediction.departure_time.departure_time_predictor import DepartureTimePredictor

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.job.job_to_vector_transformer import JobToVectorTransformer

class SklearnRegressionDepartureTimePredictor(DepartureTimePredictor):
    """
    departure time predictor that uses a simple sklearn regression. this
    neglects the dependency between jobs and only considers the features of a
    job.
    """
    def __init__(self, regressor: RegressorMixin, job_to_vector_transformer: JobToVectorTransformer):
        """
        creates a new departure time predictor

        Parameters
        ----------
        regressor : RegressorMixin
            trained regression model that supports the sklearn API, i.e. it
            should offer a predict method. the predict method should return a
            prediction for the sojourn time of a job.
        job_to_vector_transformer : JobToVectorTransformer
            used to transform the features of a job into a vector that can be understood
            by the regressor
        """
        self.__job_to_vector_transformer = job_to_vector_transformer
        self.__regressor = regressor

    def predict_waiting_and_departure_times_distribution(
            self, arrivals: list[tuple[Job, int]]) -> tuple[dict[Job, list[int]]|None, dict[Job, list[int]]]:
        X = np.array([
            self.__job_to_vector_transformer.transform(job)
            for job,_ in arrivals
        ])
        return None, {
            arrivals[i][0]: [arrivals[i][1] + predicted_sojourn_time]
            for i, predicted_sojourn_time in enumerate(self.__regressor.predict(X))
        }
