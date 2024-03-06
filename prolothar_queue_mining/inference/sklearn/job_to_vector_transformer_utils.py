from typing import Iterable

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.job.job_to_vector_transformer import JobToVectorTransformer
from prolothar_queue_mining.model.job.job_to_vector_transformer import OneHotEncoder
from prolothar_queue_mining.model.job.job_to_vector_transformer import StandardScaler
from prolothar_queue_mining.model.job.job_to_vector_transformer import MinMaxScaler

def create_job_to_vector_transformer_for_random_forest(
        jobs: Iterable[Job], categorical_features: list[str],
        numerical_features: list[str]) -> JobToVectorTransformer:
    return JobToVectorTransformer(
        numerical_features, categorical_features, {},
        {
            feature_name: OneHotEncoder.train(jobs, feature_name)
            for feature_name in categorical_features
        }
    )

def create_job_to_vector_transformer_for_linear_model(
        jobs: Iterable[Job], categorical_features: list[str],
        numerical_features: list[str]) -> JobToVectorTransformer:
    return JobToVectorTransformer(
        numerical_features, categorical_features,
        {
            feature_name: StandardScaler.train(jobs, feature_name)
            for feature_name in numerical_features
        },
        {
            feature_name: OneHotEncoder.train(jobs, feature_name)
            for feature_name in categorical_features
        }
    )

def create_job_to_vector_transformer_for_neural_model(
        jobs: Iterable[Job], categorical_features: list[str],
        numerical_features: list[str]) -> JobToVectorTransformer:
    return JobToVectorTransformer(
        numerical_features, categorical_features,
        {
            feature_name: MinMaxScaler.train(jobs, feature_name)
            for feature_name in numerical_features
        },
        {
            feature_name: OneHotEncoder.train(jobs, feature_name)
            for feature_name in categorical_features
        }
    )