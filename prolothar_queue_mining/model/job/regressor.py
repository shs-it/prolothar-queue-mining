from abc import ABC, abstractmethod

from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel
from sklearn.linear_model._glm import _GeneralizedLinearRegressor

import prolothar_common.mdl_utils as mdl_utils

from prolothar_queue_mining.model.job.job import Job
from prolothar_queue_mining.model.job.job_to_vector_transformer import JobToVectorTransformer

class Regressor(ABC):
    """
    interface for a regression function on a job
    """
    @abstractmethod
    def predict(self, job: Job) -> float:
        """
        predicts a value for a given job
        """
    @abstractmethod
    def get_mdl_of_model(self) -> float:
        """
        returns the MDL based model complexity
        """

class SklearnRegressor(Regressor):
    __slots__ = '__sklearn_model', '__job_to_vector_transformer', '__cache_enabled', '__cache'
    """
    implementation of a regressor that uses an underlying sklearn model
    """
    def __init__(
        self, sklearn_model: RegressorMixin,
        job_to_vector_transformer: JobToVectorTransformer,
        cache_enabled: bool = False):
        """
        creates a new SklearnRegressor

        Parameters
        ----------
        sklearn_model : RegressorMixin
            underlying sklearn model used to make predictions
        job_to_vector_transformer : JobToVectorTransformer
            used to create a numpy vector for a job as input to the sklearn_model
        cache_enabled : bool, optional
            if True, then all predictions are stored in a dictionary.
            only useful if one and the same job is predicted multiple times,
            by default False
        """
        self.__sklearn_model = sklearn_model
        self.__job_to_vector_transformer = job_to_vector_transformer
        self.__cache_enabled = cache_enabled
        self.__cache: dict[Job, float] = {}

    def predict(self, job: Job) -> float:
        if self.__cache_enabled:
            try:
                return self.__cache[job]
            except KeyError:
                try:
                    prediction = self.__sklearn_model.predict_single(
                        self.__job_to_vector_transformer.transform(job))
                except AttributeError:
                    prediction = self.__sklearn_model.predict(
                        self.__job_to_vector_transformer.transform(job).reshape(1, -1))[0]
                self.__cache[job] = prediction
                return prediction
        else:
            return self.__sklearn_model.predict(
                self.__job_to_vector_transformer.transform(job).reshape(1, -1))[0]

    def get_mdl_of_model(self) -> float:
        mdl_of_model = 0
        try:
            for c in self.__sklearn_model.coef_:
                mdl_of_model += mdl_utils.L_R(c)
        except AttributeError:
            pass
        try:
            mdl_of_model += mdl_utils.L_R(self.__sklearn_model.intercept_)
        except AttributeError:
            pass
        return mdl_of_model

    def __repr__(self):
        if isinstance(self.__sklearn_model, LinearModel) \
        or isinstance(self.__sklearn_model, _GeneralizedLinearRegressor):
            weights_dict = {
                feature: coefficient for feature, coefficient in zip(
                    self.__job_to_vector_transformer.get_feature_names_of_vector_components(),
                    self.__sklearn_model.coef_)
                if coefficient != 0
            }
            weights_dict['intercept'] = self.__sklearn_model.intercept_
            return f'{self.__sklearn_model.__class__.__name__}({weights_dict})'
        else:
            return repr(self.__sklearn_model)
