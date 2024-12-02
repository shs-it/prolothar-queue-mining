import sys
from tqdm import trange

from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.arrival_process import FixedArrival
from prolothar_queue_mining.model.job import Job

from prolothar_queue_mining.inference.queue.nr_of_servers import LowerBoundEstimator
from prolothar_queue_mining.inference.queue.nr_of_servers import UpperBoundEstimator

from prolothar_queue_mining.inference.queue.cuemin.search_strategy.search_strategy import SearchStrategy

class LinearSearch(SearchStrategy):

    def __init__(self,
        patience: int = sys.maxsize,
        recording_enabled: bool = True,
        record_candidates: bool = False,
        verbose: bool = False,
        seed_for_distributions: int = None,
        nr_of_cpus_for_sklearn: int = 1,
        nr_of_load_clusters_candidates: list[int] = None,
        categorical_attribute_names: list[str] = None,
        numerical_attribute_names: list[str] = None,
        min_nr_of_servers: int|None = None,
        max_nr_of_servers: int|None = None):
        super().__init__(
            record_candidates=record_candidates,
            recording_enabled=recording_enabled,
            verbose=verbose,
            seed_for_distributions=seed_for_distributions,
            nr_of_cpus_for_sklearn=nr_of_cpus_for_sklearn,
            nr_of_load_clusters_candidates=nr_of_load_clusters_candidates,
            categorical_attribute_names=categorical_attribute_names,
            numerical_attribute_names=numerical_attribute_names)
        self.__patience = patience
        self.__min_nr_of_servers = min_nr_of_servers
        self.__max_nr_of_servers = max_nr_of_servers

    def search(
            self, waiting_area: WaitingArea,
            observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]],
            departure_time_per_job: dict[Job, int],
            arrival_process: FixedArrival,
            nr_of_jobs_in_system_over_time: list[int]) -> tuple[Queue, float]:
        iterations_without_improvement = 0
        best_mdl_score = float('inf')
        best_queue = None
        if self.__min_nr_of_servers is None:
            min_c = LowerBoundEstimator(waiting_area).estimate_nr_of_servers(observed_arrivals, observed_departures)
        else:
            min_c = self.__min_nr_of_servers
        if self.__max_nr_of_servers is None:
            max_c = UpperBoundEstimator(waiting_area).estimate_nr_of_servers(observed_arrivals, observed_departures)
        else:
            max_c = self.__max_nr_of_servers
        for nr_of_servers in trange(min_c, max_c + 1, disable=not self.verbose, desc='c', leave=False):
            best_queue_for_c, best_mdl_score_for_c = self._find_best_queue_for_c(
                observed_arrivals, observed_departures, departure_time_per_job,
                nr_of_jobs_in_system_over_time, arrival_process,
                waiting_area, nr_of_servers)

            if best_mdl_score_for_c < best_mdl_score:
                iterations_without_improvement = 0
                best_mdl_score = best_mdl_score_for_c
                best_queue = best_queue_for_c
            else:
                iterations_without_improvement += 1
                if iterations_without_improvement > self.__patience and nr_of_servers > min_c:
                    break
        return best_queue, best_mdl_score