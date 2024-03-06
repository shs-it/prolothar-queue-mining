from typing import Iterable
from sklearn.linear_model import LogisticRegression

from prolothar_queue_mining.inference.queue.waiting_area.pairwise_sklearn_cv_estimator import PairwiseSklearnCvEstimator
from prolothar_queue_mining.model.job.job_to_vector_transformer import JobToVectorTransformer

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.job.job_to_vector_transformer import MinMaxScaler
from prolothar_queue_mining.model.job.job_to_vector_transformer import OneHotEncoder

class PairwiseLogisticRegressionCvEstimator(PairwiseSklearnCvEstimator):

    def __init__(self, numerical_feature_names: str, categorical_feature_names: str):
        super().__init__(
            numerical_feature_names, categorical_feature_names,
            LogisticRegression(), {'C': [0.1, 1, 10]})

    def _create_job_to_vector_transformer(
            self, numerical_feature_names: list[str], categorical_feature_names: list[str],
            jobs: Iterable[Job]):
        return JobToVectorTransformer(
            numerical_feature_names,
            categorical_feature_names,
            {
                feature_name: MinMaxScaler.train(jobs, feature_name)
                for feature_name in numerical_feature_names
            },
            {
                feature_name: OneHotEncoder.train(jobs, feature_name)
                for feature_name in categorical_feature_names
            },
        )

