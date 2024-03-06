from abc import ABC, abstractmethod
from typing import Iterable
import numpy as np
from prolothar_common.experiments.statistics import Statistics

from prolothar_queue_mining.model.job import Job

class Scaler(ABC):
    """
    interface of a scaler for numerical features
    """
    @abstractmethod
    def scale(self, value: float) -> float:
        """
        scales the given value
        """

class NoScaler(Scaler):
    """
    dummy scaler that does nothing
    """
    def scale(self, value: float) -> float:
        return value

class MinMaxScaler(Scaler):
    """
    a scaler that scales all values between 0 and 1
    """
    def __init__(self, min_value: float, max_value: float):
        self.min_value = min_value
        self.value_range = max_value - min_value

    def scale(self, value: float) -> float:
        return (value - self.min_value) / self.value_range

    @staticmethod
    def train(job_list: Iterable[Job], feature_name: str):
        scaler = MinMaxScaler(
            min(job.features[feature_name] for job in job_list),
            max(job.features[feature_name] for job in job_list)
        )
        if scaler.value_range != 0:
            return scaler
        else:
            return NoScaler()

class StandardScaler(Scaler):
    """
    scaler that removes mean brings value to unit variance
    """
    def __init__(self, mean: float, standard_deviation: float):
        self.mean = mean
        self.standard_deviation = standard_deviation

    def scale(self, value: float) -> float:
        return (value - self.mean) / self.standard_deviation

    @staticmethod
    def train(job_list: Iterable[Job], feature_name: str):
        statistics = Statistics(job.features[feature_name] for job in job_list)
        if statistics.stddev() > 0:
            return StandardScaler(statistics.mean(), statistics.stddev())
        else:
            return NoScaler()

class Encoder(ABC):
    """
    interface of an encoder for categorical features
    """
    @abstractmethod
    def encode(self, value) -> tuple[int]:
        """
        encodes the given value
        """

    @abstractmethod
    def get_derived_feature_names(self, original_feature_name: str) -> list[str]:
        """
        provides human-readable names for the derived features by this encoder
        """

class NoEncoder(Encoder):
    """
    dummy encoder that does nothing
    """
    def encode(self, value) -> tuple[int]:
        return (value,)

    def get_derived_feature_names(self, original_feature_name: str) -> list[str]:
        return [original_feature_name]

class OneHotEncoder(Encoder):
    """
    One-Hot-Encoder
    """
    def __init__(self, possible_values: list):
        self.possible_values = possible_values

    def encode(self, value) -> tuple[int]:
        return (1 if value == category else 0 for category in self.possible_values)

    def get_derived_feature_names(self, original_feature_name: str) -> list[str]:
        return [f'{original_feature_name} = {category}' for category in self.possible_values]

    @staticmethod
    def train(job_list: Iterable[Job], feature_name: str):
        return OneHotEncoder(set(job.features[feature_name] for job in job_list))

class JobToVectorTransformer():
    """
    transformer that creates a numpy array from the features of a given job
    """
    def __init__(
        self, numerical_features: list[str], categorical_features: list[str],
        scalers: dict[str, Scaler], encoders: dict[str, Encoder]):
        """
        creates a new transformer instance

        Parameters
        ----------
        numerical_features : list[str]
            determines which numerical features a part of the feature vector and in
            which order.
        categorical_features : list[str]
            determines which categorical features a part of the feature vector and in
            which order
        scalers : dict[str, Scaler]
            defines the usage of scalers (e.g. min-max-scaling) for numerical features.
        encoders : dict[str, Encoder]
            defines the usage of encoders (e.g. one-hot-encoding) for categorical features
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.scalers = scalers
        self.encoders = encoders

    def transform(self, job: Job) -> np.ndarray:
        """
        transforms the given job to a numpy array
        """
        return np.array(
            [
                self.scalers.get(feature_name, NoScaler()).scale(job.features[feature_name])
                for feature_name in self.numerical_features
            ] +
            [
                encoded_value for feature_name in self.categorical_features for encoded_value
                in self.encoders.get(feature_name, NoEncoder()).encode(job.features[feature_name])
            ]
        )

    def get_feature_names_of_vector_components(self) -> list[str]:
        feature_names = list(self.numerical_features)
        for feature_name in self.categorical_features:
            feature_names.extend(
                self.encoders.get(feature_name, NoEncoder()).get_derived_feature_names(feature_name)
            )
        return feature_names
