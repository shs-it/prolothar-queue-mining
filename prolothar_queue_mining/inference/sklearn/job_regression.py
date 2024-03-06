"""
provides utility functions to train sklearn regressors based on the features
of Jobs
"""

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from prolothar_queue_mining.sklearn_barebones.linear_model import BarebonesRidge as Ridge
from prolothar_queue_mining.sklearn_barebones.linear_model import BarebonesLasso
from prolothar_queue_mining.sklearn_barebones.linear_model import BarebonesGammaRegressor as GammaRegressor

from prolothar_queue_mining.model.distribution import NormalDistribution
from prolothar_queue_mining.model.distribution import TwoSidedGeometricDistribution
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.job.job_to_vector_transformer import JobToVectorTransformer


from prolothar_queue_mining.inference.sklearn.job_to_vector_transformer_utils import create_job_to_vector_transformer_for_random_forest
from prolothar_queue_mining.inference.sklearn.job_to_vector_transformer_utils import create_job_to_vector_transformer_for_linear_model

def train_random_forest_cv(
        y_per_job: dict[Job, int],
        categorical_features: list[str],
        numerical_features: list[str],
        nr_of_folds=5) -> tuple[RandomForestRegressor, JobToVectorTransformer]:
    param_grid = {
        'n_estimators': [5, 25, 50, 75, 100],
        'max_depth': [2, 5, 7, 9],
        'min_samples_leaf': [1, 10, 50]
    }
    model = GridSearchCV(RandomForestRegressor(), param_grid, cv=nr_of_folds)
    job_to_vector_transformer = create_job_to_vector_transformer_for_random_forest(
        y_per_job.keys(), categorical_features, numerical_features)
    X = np.array([
        job_to_vector_transformer.transform(job)
        for job,_ in y_per_job.items()
    ])
    y = np.array([y for _,y in y_per_job.items()])

    model.fit(X, y)
    return model.best_estimator_, job_to_vector_transformer

def train_ridge_regression_cv(
        y_per_job: dict[Job, int],
        categorical_features: list[str],
        numerical_features: list[str],
        nr_of_folds=5) -> tuple[Ridge, JobToVectorTransformer, NormalDistribution]:
    param_grid = {
        'alpha': [0.01, 0.1, 1.0, 10],
    }
    model = GridSearchCV(Ridge(), param_grid, cv=nr_of_folds)
    job_to_vector_transformer = create_job_to_vector_transformer_for_linear_model(
        y_per_job.keys(), categorical_features, numerical_features)
    X = np.array([
        job_to_vector_transformer.transform(job)
        for job,_ in y_per_job.items()
    ])
    y = np.array([y for _,y in y_per_job.items()])

    model.fit(X, y)

    residuals = y - model.best_estimator_.predict(X)
    error_distribution = NormalDistribution.fit(residuals)
    return model.best_estimator_, job_to_vector_transformer, error_distribution

def train_lasso_cv(
        y_per_job: dict[Job, int],
        categorical_features: list[str],
        numerical_features: list[str],
        force_positive_coefficients: bool = False,
        nr_of_folds=5,
        nr_of_cpus: int = 1) -> tuple[Ridge, JobToVectorTransformer, NormalDistribution]:
    param_grid = {
        'alpha': [0.01, 0.1, 1.0, 10],
    }
    model = GridSearchCV(
        BarebonesLasso(positive=force_positive_coefficients),
        param_grid, cv=nr_of_folds, n_jobs=nr_of_cpus)
    job_to_vector_transformer = create_job_to_vector_transformer_for_linear_model(
        y_per_job.keys(), categorical_features, numerical_features)
    X = np.array([
        job_to_vector_transformer.transform(job)
        for job,_ in y_per_job.items()
    ])
    y = np.array([y for _,y in y_per_job.items()])

    model.fit(X, y)

    residuals = y - model.best_estimator_.predict(X)
    error_distribution = NormalDistribution.fit(residuals)
    return model.best_estimator_, job_to_vector_transformer, error_distribution

def train_gamma_regression_cv(
        y_per_job: dict[Job, int],
        categorical_features: list[str],
        numerical_features: list[str],
        nr_of_folds: int = 5,
        nr_of_cpus: int = 1) -> tuple[GammaRegressor, JobToVectorTransformer, TwoSidedGeometricDistribution]:
    param_grid = {
        'alpha': [0.01, 0.1, 1.0, 10],
    }
    model = GridSearchCV(GammaRegressor(), param_grid, cv=nr_of_folds, n_jobs=nr_of_cpus)
    job_to_vector_transformer = create_job_to_vector_transformer_for_linear_model(
        y_per_job.keys(), categorical_features, numerical_features)
    X = np.array([
        job_to_vector_transformer.transform(job)
        for job,_ in y_per_job.items()
    ])
    y = np.array([y for _,y in y_per_job.items()])

    model.fit(X, y)

    residuals = y - model.best_estimator_.predict(X)
    error_distribution = TwoSidedGeometricDistribution.fit(residuals)
    return model.best_estimator_, job_to_vector_transformer, error_distribution

