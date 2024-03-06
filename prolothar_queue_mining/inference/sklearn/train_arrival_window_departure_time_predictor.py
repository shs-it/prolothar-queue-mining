from itertools import chain
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.prediction.departure_time import ArrivalWindowDepartureTimePredictor
from prolothar_queue_mining.inference.sklearn.job_to_vector_transformer_utils import create_job_to_vector_transformer_for_random_forest
from prolothar_queue_mining.inference.sklearn.job_to_vector_transformer_utils import create_job_to_vector_transformer_for_linear_model

def train_random_forest_cv(
        arrivals: list[tuple[Job, int]],
        departures: list[tuple[Job, int]],
        categorical_features: list[str],
        numerical_features: list[str],
        past_arrivals: int = 10,
        future_arrivals: int = 10,
        nr_of_folds: int = 5) -> ArrivalWindowDepartureTimePredictor:
    all_jobs = set(map(lambda x: x[0], chain(arrivals, departures)))

    job_to_vector_transformer = create_job_to_vector_transformer_for_random_forest(
        all_jobs, categorical_features, numerical_features)

    param_grid = {
        'n_estimators': [5, 25, 50, 75, 100],
        'max_depth': [2, 5, 7, 9],
        'min_samples_leaf': [1, 10, 50]
    }

    predictor = ArrivalWindowDepartureTimePredictor(
        past_arrivals, future_arrivals, job_to_vector_transformer,
        GridSearchCV(RandomForestRegressor(), param_grid, cv=nr_of_folds))

    predictor.train_regression_model(arrivals, departures)

    return predictor

def train_decision_tree_cv(
        arrivals: list[tuple[Job, int]],
        departures: list[tuple[Job, int]],
        categorical_features: list[str],
        numerical_features: list[str],
        past_arrivals: int = 10,
        future_arrivals: int = 10,
        nr_of_folds: int = 5) -> ArrivalWindowDepartureTimePredictor:
    all_jobs = set(map(lambda x: x[0], chain(arrivals, departures)))

    job_to_vector_transformer = create_job_to_vector_transformer_for_random_forest(
        all_jobs, categorical_features, numerical_features)

    param_grid = {
        'criterion': ['squared_error', 'absolute_error'],
        'max_depth': [2, 5, 7, 9],
        'min_samples_leaf': [1, 10, 50]
    }

    predictor = ArrivalWindowDepartureTimePredictor(
        past_arrivals, future_arrivals, job_to_vector_transformer,
        GridSearchCV(DecisionTreeRegressor(), param_grid, cv=nr_of_folds))

    predictor.train_regression_model(arrivals, departures)

    return predictor

def train_linear_regression(
        arrivals: list[tuple[Job, int]],
        departures: list[tuple[Job, int]],
        categorical_features: list[str],
        numerical_features: list[str],
        past_arrivals: int = 10,
        future_arrivals: int = 10) -> ArrivalWindowDepartureTimePredictor:
    all_jobs = set(map(lambda x: x[0], chain(arrivals, departures)))

    job_to_vector_transformer = create_job_to_vector_transformer_for_linear_model(
        all_jobs, categorical_features, numerical_features)

    predictor = ArrivalWindowDepartureTimePredictor(
        past_arrivals, future_arrivals, job_to_vector_transformer,
        LinearRegression())

    predictor.train_regression_model(arrivals, departures)

    return predictor

