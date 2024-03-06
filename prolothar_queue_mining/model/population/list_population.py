from prolothar_queue_mining.model.population.population import Population
from prolothar_queue_mining.model.job import Job

class ListPopulation(Population):
    """
    a population that just consists of a static list. the jobs in this population
    are yield in the order of the given list. if all jobs have been yielded, there
    will no further jobs be available during simulation
    """
    def __init__(self, job_list: list[Job], initial_job_index: int = 0):
        self.__job_list = job_list
        self.__current_job_index = initial_job_index
        self.__nr_of_jobs = len(job_list)

    def get_next_job(self) -> Job:
        if self.__current_job_index < self.__nr_of_jobs:
            next_job = self.__job_list[self.__current_job_index]
            self.__current_job_index += 1
            return next_job
        else:
            raise StopIteration()

    def get_ith_job(self, i: int) -> Job:
        return self.__job_list[i]

    def get_job_list(self) -> list[Job]:
        return self.__job_list

    def copy(self) -> Population:
        return ListPopulation(self.__job_list, initial_job_index = self.__current_job_index)

    def set_seed(self, seed: int):
        #no action required, because no randomness included
        pass
