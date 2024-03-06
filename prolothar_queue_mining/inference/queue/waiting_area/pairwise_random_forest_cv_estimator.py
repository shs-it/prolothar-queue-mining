from typing import Iterable
from sklearn.ensemble import RandomForestClassifier

from prolothar_queue_mining.inference.queue.waiting_area.pairwise_sklearn_cv_estimator import PairwiseSklearnCvEstimator
from prolothar_queue_mining.inference.sklearn.job_to_vector_transformer_utils import create_job_to_vector_transformer_for_random_forest

from prolothar_queue_mining.model.job import Job

class PairwiseRandomForestCvEstimator(PairwiseSklearnCvEstimator):

    def __init__(self, numerical_feature_names: str, categorical_feature_names: str):
        super().__init__(
            numerical_feature_names, categorical_feature_names,
            RandomForestClassifier(),
            {
                'n_estimators': [5, 50, 100],
                'max_depth': [2, 5, 7, 9],
                'min_samples_leaf': [1, 10, 50]
            }
        )

    def _create_job_to_vector_transformer(
            self, numerical_feature_names: list[str], categorical_feature_names: list[str],
            jobs: Iterable[Job]):
        return create_job_to_vector_transformer_for_random_forest(
            jobs, categorical_feature_names, numerical_feature_names)
