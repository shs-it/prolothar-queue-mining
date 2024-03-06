from numpy.random import default_rng

from prolothar_queue_mining.model.population import Population
from prolothar_queue_mining.model.arrival_process.arrival_process import ArrivalProcess
from prolothar_queue_mining.model.job import Job

class ExponentialDistributedArrival(ArrivalProcess):
    """
    "M" in Kendall's notation
    """

    def __init__(
            self, population: Population, mean_arrival_rate: float, seed: int|None = None,
            last_arrival: int = 0):
        self.__population = population
        self.__mean_arrival_rate = mean_arrival_rate
        self.__random_number_generator = default_rng(seed)
        self.__last_arrival = last_arrival
        self.__seed = seed

    def get_next_job(self) -> tuple[int,Job]:
        inter_arrival_time = round(self.__random_number_generator.exponential(1 / self.__mean_arrival_rate))
        self.__last_arrival += inter_arrival_time
        return self.__last_arrival, self.__population.get_next_job()

    def get_mean_arrival_rate(self) -> float:
        return self.__mean_arrival_rate

    def copy(self) -> ArrivalProcess:
        return ExponentialDistributedArrival(
            self.__population.copy(), self.__mean_arrival_rate,
            seed=self.__seed, last_arrival=self.__last_arrival)

    def set_seed(self, seed: int):
        self.__population.set_seed(seed)
        self.__seed = seed

    def __repr__(self):
        return f'Exponential({self.__mean_arrival_rate})'
