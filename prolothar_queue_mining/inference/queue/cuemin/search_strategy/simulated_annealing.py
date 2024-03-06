from random import Random

from simanneal import Annealer
from methodtools import lru_cache
from tqdm import tqdm

from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.arrival_process import FixedArrival
from prolothar_queue_mining.model.job import Job

from prolothar_queue_mining.inference.queue.nr_of_servers import UpperBoundEstimator

from prolothar_queue_mining.inference.queue.cuemin.search_strategy.search_strategy import SearchStrategy

class SimulatedAnnealing(SearchStrategy, Annealer):

    def __init__(self,
        recording_enabled: bool = True,
        record_candidates: bool = False,
        nr_of_iterations: int = 200,
        verbose: bool = False,
        seed_for_distributions: int = None,
        nr_of_cpus_for_sklearn: int = 1,
        nr_of_load_clusters_candidates: list[int] = None,
        categorical_attribute_names: list[str] = None,
        numerical_attribute_names: list[str] = None):
        SearchStrategy.__init__(self,
            record_candidates=record_candidates,
            recording_enabled=recording_enabled,
            verbose=verbose,
            seed_for_distributions=seed_for_distributions,
            nr_of_cpus_for_sklearn=nr_of_cpus_for_sklearn,
            nr_of_load_clusters_candidates=nr_of_load_clusters_candidates,
            categorical_attribute_names=categorical_attribute_names,
            numerical_attribute_names=numerical_attribute_names)
        Annealer.__init__(self, 1)
        self.__random = Random(seed_for_distributions)
        self.steps = nr_of_iterations
        self.verbose = verbose

    def move(self):
        if self.state == 1:
            self.state = 2
        elif self.state == self.__max_c or self.__random.random() < 0.5:
            self.state -= 1
        else:
            self.state += 1

    def energy(self):
        return self.__compute_mdl_for_c(self.state)

    @lru_cache(maxsize=-1)
    def __compute_mdl_for_c(self, c: int) -> float:
        best_queue_for_c, best_mdl_score_for_c = self._find_best_queue_for_c(
            self.__observed_arrivals,
            self.__observed_departures,
            self.__departure_time_per_job,
            self.__nr_of_jobs_in_system_over_time,
            self.__arrival_process,
            self.__waiting_area, c)
        if best_mdl_score_for_c < self.__best_mdl_score:
            self.__best_queue = best_queue_for_c
            self.__best_mdl_score = best_mdl_score_for_c
        self.__visited_states.add(c)
        if len(self.__visited_states) == self.__max_c:
            self.set_user_exit(None, None)
        return best_mdl_score_for_c

    def update(self, step, T, E, acceptance, improvement):
        self.__progress_bar.update(step)

    def search(
            self, waiting_area: WaitingArea,
            observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]],
            departure_time_per_job: dict[Job, int],
            arrival_process: FixedArrival,
            nr_of_jobs_in_system_over_time: list[int]) -> tuple[Queue, float]:
        self.__max_c = UpperBoundEstimator(waiting_area).estimate_nr_of_servers(observed_arrivals, observed_departures)
        self.__waiting_area = waiting_area
        self.__observed_arrivals = observed_arrivals
        self.__observed_departures = observed_departures
        self.__departure_time_per_job = departure_time_per_job
        self.__arrival_process = arrival_process
        self.__nr_of_jobs_in_system_over_time = nr_of_jobs_in_system_over_time
        self.__best_mdl_score = float('inf')
        self.__visited_states = set()
        try:
            self.__progress_bar = tqdm(total=self.steps, disable=not self.verbose)
            self.anneal()
        finally:
            self.__progress_bar.close()
        return self.__best_queue, self.__best_mdl_score