from prolothar_queue_mining.model.population import Population
from prolothar_queue_mining.model.arrival_process.arrival_process import ArrivalProcess
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.distribution import Distribution

class ArrivalWithDistribution(ArrivalProcess):
    """
    "M" in Kendall's notation
    """

    def __init__(
            self, population: Population, distribution: Distribution,
            last_arrival: int = 0):
        self.__population = population
        self.__distribution = distribution
        self.__last_arrival = last_arrival

    def get_next_job(self) -> tuple[int,Job]:
        inter_arrival_time = round(self.__distribution.get_next_sample())
        self.__last_arrival += inter_arrival_time
        return self.__last_arrival, self.__population.get_next_job()

    def get_mean_arrival_rate(self) -> float:
        return 1 / self.__distribution.get_mean()

    def copy(self) -> ArrivalProcess:
        return ArrivalWithDistribution(
            self.__population.copy(), self.__distribution.copy(),
            last_arrival=self.__last_arrival)

    def set_seed(self, seed: int):
        self.__population.set_seed(seed)
        self.__distribution.set_seed(seed)

    def __repr__(self):
        return f'ArrivalWithDistribution({self.__distribution})'
