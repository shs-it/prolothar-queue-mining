from prolothar_queue_mining.model.population import Population
from prolothar_queue_mining.model.arrival_process.arrival_process import ArrivalProcess
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.population import ListPopulation

class FixedArrival(ArrivalProcess):

    def __init__(self, population: Population, arrival_times: list[int], initial_index: int = 0):
        self.__population = population
        self.__arrival_times = arrival_times
        self.__current_index = initial_index
        self.__nr_of_arrivals = len(self.__arrival_times)

    def get_next_job(self) -> tuple[int,Job]:
        if self.has_next_job():
            arrival_time, job = self.__arrival_times[self.__current_index], self.__population.get_next_job()
            self.__current_index += 1
            return arrival_time, job
        else:
            raise StopIteration()

    def has_next_job(self) -> bool:
        return self.__current_index < self.__nr_of_arrivals

    def get_ith_arrival_time(self, i: int) -> int:
        return self.__arrival_times[i]

    def get_mean_arrival_rate(self) -> float:
        return (self.__nr_of_arrivals - 1) / (self.__arrival_times[-1] - self.__arrival_times[0])

    def copy(self) -> ArrivalProcess:
        return FixedArrival(
            self.__population.copy(), self.__arrival_times, initial_index=self.__current_index)

    def set_seed(self, seed: int):
        self.__population.set_seed(seed)

    def __repr__(self):
        return 'FixedArrival(...)'

    @staticmethod
    def create_from_observation(observed_arrivals: list[tuple[Job, int]]) -> 'FixedArrival':
        return FixedArrival(
            ListPopulation([job for job, _ in observed_arrivals]),
            [arrival_time for _, arrival_time in observed_arrivals])
