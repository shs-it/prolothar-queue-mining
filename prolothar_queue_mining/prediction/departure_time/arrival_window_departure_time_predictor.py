import numpy as np
from sklearn.base import RegressorMixin

from prolothar_queue_mining.prediction.departure_time.departure_time_predictor import DepartureTimePredictor

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.job.job_to_vector_transformer import JobToVectorTransformer

class ArrivalWindowDepartureTimePredictor(DepartureTimePredictor):
    """
    a regressor that predicts departure time for each job by learning a regression
    model with the attributes being the relative arrival times and job attributes
    of a window around the job for which we want to predict the departure time
    """
    def __init__(
            self, past_arrivals: int, future_arrivals: int,
            job_to_vector_transformer: JobToVectorTransformer,
            regression_model: RegressorMixin):
        self.__past_arrivals = past_arrivals
        self.__future_arrivals = future_arrivals
        self.__job_to_vector_transformer = job_to_vector_transformer
        self.__regression_model = regression_model

    def get_regression_model(self) -> RegressorMixin:
        return self.__regression_model

    def train_regression_model(self, arrivals: list[tuple[Job, int]], departures: list[tuple[Job, int]]):
        """
        retrains the underlying regression model

        Parameters
        ----------
        arrivals : list[tuple[Job, int]]
            list of training arrival = jobs + corresponding arrival time
        departures : list[tuple[Job, int]]
            list of training departures = jobs + corresponding departure time.
            used to compute training sojourn times, which is the target of
            the regression
        """
        departure_time_per_job = dict(departures)
        X = []
        y = []
        for i in range(len(arrivals)):
            job, arrival_time = arrivals[i]
            if job in departure_time_per_job:
                X.append(self.__create_feature_vector(arrivals, i))
                y.append(departure_time_per_job[job] - arrival_time)

        X = np.vstack(X)
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])

        y = np.array(y)

        self.__regression_model.fit(X, y)

    def __create_feature_vector(self, arrivals: list[tuple[Job, int]], i: int):
        job, arrival_time = arrivals[i]
        job_vector = self.__job_to_vector_transformer.transform(job)
        feature_vector = []
        for _ in range(self.__past_arrivals - i):
            feature_vector.extend([np.nan] * (1 + len(job_vector)))
        for j in range(max(0, i - self.__past_arrivals), i):
            previous_job, previous_arrival_time = arrivals[j]
            feature_vector.extend(self.__job_to_vector_transformer.transform(previous_job))
            feature_vector.append(previous_arrival_time - arrival_time)
        feature_vector.extend(self.__job_to_vector_transformer.transform(job))
        for j in range(i+1, min(len(arrivals), i+1+self.__future_arrivals)):
            next_job, next_arrival_time = arrivals[j]
            feature_vector.extend(self.__job_to_vector_transformer.transform(next_job))
            feature_vector.append(next_arrival_time - arrival_time)
        for _ in range(self.__future_arrivals - len(arrivals) + i + 1):
            feature_vector.extend([np.nan] * (1 + len(job_vector)))
        return np.array(feature_vector)

    def  get_feature_names_for_vector(self) -> list[str]:
        feature_names = []
        non_temporal_features_names = self.__job_to_vector_transformer.get_feature_names_of_vector_components()
        for i in range(self.__past_arrivals):
            for feature_name in non_temporal_features_names:
                feature_names.append(f'{feature_name} (-{self.__past_arrivals - i})')
            feature_names.append(f'relative arrival (-{self.__past_arrivals - i})')
        for feature_name in non_temporal_features_names:
            feature_names.append(f'{feature_name} (0)')
        for i in range(self.__future_arrivals):
            for feature_name in non_temporal_features_names:
                feature_names.append(f'{feature_name} ({i+1})')
            feature_names.append(f'relative arrival ({i+1})')

        return feature_names

    def predict_waiting_and_departure_times_distribution(
            self, arrivals: list[tuple[Job, int]]) -> tuple[dict[Job, list[int]] | None, dict[Job, list[int]]]:
        X = []
        for i in range(len(arrivals)):
            X.append(self.__create_feature_vector(arrivals, i))
        X = np.vstack(X)
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])

        y = self.__regression_model.predict(X)

        return None, {
            arrivals[i][0]: [arrivals[i][1] + y[i]]
            for i in range(len(y))
        }
