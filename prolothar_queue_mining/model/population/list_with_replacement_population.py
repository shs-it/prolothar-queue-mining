from random import Random

from prolothar_queue_mining.model.population.population import Population
from prolothar_queue_mining.model.job import Job

class ListWithReplacementPopulation(Population):
    """
    the jobs in this population are drawn from a finite list with replacement.
    note that the job id will be overriden to ensure unique job ids.
    """
    def __init__(self, job_list: list[Job], seed: int = None):
        self.__job_list = job_list
        self.set_seed(seed)
        self.__nr_of_returned_jobs = 0

    def get_next_job(self) -> Job:
        job_id = str(self.__nr_of_returned_jobs)
        self.__nr_of_returned_jobs += 1
        return Job(job_id, self.__random.choice(self.__job_list).features)

    def get_job_list(self) -> list[Job]:
        return self.__job_list

    def copy(self) -> Population:
        return ListWithReplacementPopulation(self.__job_list, seed=self.__seed)

    def set_seed(self, seed: int):
        self.__seed = seed
        self.__random = Random(seed)
