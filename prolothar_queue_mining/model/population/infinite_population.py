from random import Random

from prolothar_queue_mining.model.population.population import Population
from prolothar_queue_mining.model.job import Job

class InfinitePopulation(Population):
    """
    mainly for testing purpose. a population that can generate (in theory) infinitely many
    jobs with incrementing job ID and optionally also with randomly generated features
    """
    def __init__(self, initial_job_id: int = 0,
                 categorical_feature_names: list[str]=None,
                 numerical_feature_names: list[str]=None,
                 nr_of_categories: int = 3,
                 seed: int = None):
        self.__next_job_id = initial_job_id
        if categorical_feature_names is not None:
            self.__categorical_feature_names = categorical_feature_names
        else:
            self.__categorical_feature_names = []
        if numerical_feature_names is not None:
            self.__numerical_feature_names = numerical_feature_names
        else:
            self.__numerical_feature_names = []
        self.__categories = list(range(nr_of_categories))
        self.set_seed(seed)


    def get_categorical_feature_names(self) -> list[str]:
        return self.__categorical_feature_names

    def get_numerical_feature_names(self) -> list[str]:
        return self.__numerical_feature_names

    def get_next_job(self) -> Job:
        job = Job(str(self.__next_job_id))
        for categorical_feature in self.__categorical_feature_names:
            job.features[categorical_feature] = self.__random.choice(self.__categories)
        for numerical_feature in self.__numerical_feature_names:
            job.features[numerical_feature] = self.__random.uniform(0, 1)
        self.__next_job_id += 1
        return job

    def copy(self) -> Population:
        return InfinitePopulation(
            initial_job_id = self.__next_job_id,
            categorical_feature_names=self.__categorical_feature_names,
            numerical_feature_names=self.__numerical_feature_names,
            nr_of_categories=len(self.__categories),
            seed=self.__seed)

    def set_seed(self, seed: int):
        self.__random = Random(seed)
        self.__seed = seed
