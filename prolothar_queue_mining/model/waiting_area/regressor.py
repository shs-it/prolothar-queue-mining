from abc import ABC, abstractmethod
import numpy as  np

from prolothar_common.mdl_utils import L_R

from prolothar_queue_mining.model.waiting_area.priority_queue import PriorityQueue
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.job.job_to_vector_transformer import JobToVectorTransformer

class Regressor(ABC):
    @abstractmethod
    def predict(self, row_vector: np.ndarray) -> float:
        """
        computes the priority of a job in row_vector form
        """
    @abstractmethod
    def describe(self, feature_names: list[str]) -> str:
        """
        returns a human-readable description of this regressor
        """
    @abstractmethod
    def get_mdl(self) -> float:
        """
        computes L(M) in number of bits
        """

class LinearRegressor(Regressor):
    def __init__(self, weights: np.ndarray):
        self.__weights = weights
    def predict(self, row_vector: np.ndarray) -> float:
        return np.matmul(self.__weights, row_vector)
    def describe(self, feature_names: list[str]) -> str:
        weights_dict = dict(zip(feature_names, self.__weights))
        weights_dict['arrival_time'] = self.__weights[-1]
        return f'LinearRegression({weights_dict})'
    def get_mdl(self) -> float:
        mdl = 0
        for weight in self.__weights:
            mdl += L_R(weight)
        return mdl

class RegressorWaitingArea(PriorityQueue):
    """
    a priority waiting area where the priority (lower means first) is computed
    by a regressor function
    """

    def __init__(self, job_to_vector_transformer: JobToVectorTransformer, regressor: Regressor):
        super().__init__()
        self.__job_to_vector_transformer = job_to_vector_transformer
        self.__regressor = regressor

    def _compute_priority(self, arrival_time: int, job: Job) -> float:
        return self.__regressor.predict(
            np.hstack((self.__job_to_vector_transformer.transform(job), arrival_time))
        )

    def get_discipline_name(self) -> str:
        feature_names = self.__job_to_vector_transformer.get_feature_names_of_vector_components()
        return f'PR({self.__regressor.describe(feature_names)})'

    def get_mdl(self, nr_of_categorical_features: int) -> float:
        return self.__regressor.get_mdl()

    def copy(self) -> 'RegressorWaitingArea':
        """
        returns a deep copy of this waiting area with the same state.
        if an attribute of this waiting area uses a random seed and the
        random seed is "None", this will result in different behavior during
        simulation. this is not a bug, but the reason for this method.
        otherwise use the "deepcopy" python module.
        """
        if len(self) > 0:
            raise NotImplementedError()
        return self.copy_empty()

    def copy_empty(self) -> 'WaitingArea':
        return RegressorWaitingArea(self.__job_to_vector_transformer, self.__regressor)
