import sys
from tqdm import trange

from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.arrival_process import FixedArrival
from prolothar_queue_mining.model.job import Job

from prolothar_queue_mining.inference.queue.cuemin.search_strategy.search_strategy import SearchStrategy
from prolothar_queue_mining.inference.queue.nr_of_servers import UpperBoundEstimator

class AdaptiveStepSizeSearch(SearchStrategy):

    def __init__(self,
        patience: int = 1,
        recording_enabled: bool = True,
        record_candidates: bool = False,
        verbose: bool = False,
        seed_for_distributions: int = None,
        nr_of_cpus_for_sklearn: int = 1,
        nr_of_load_clusters_candidates: list[int] = None,
        categorical_attribute_names: list[str] = None,
        numerical_attribute_names: list[str] = None):
        super().__init__(
            record_candidates=record_candidates,
            recording_enabled=recording_enabled,
            verbose=verbose,
            seed_for_distributions=seed_for_distributions,
            nr_of_cpus_for_sklearn=nr_of_cpus_for_sklearn,
            nr_of_load_clusters_candidates=nr_of_load_clusters_candidates,
            categorical_attribute_names=categorical_attribute_names,
            numerical_attribute_names=numerical_attribute_names)
        if patience == sys.maxsize:
            self.__patience = 10
        else:
            self.__patience = patience

    def search(
            self, waiting_area: WaitingArea,
            observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]],
            departure_time_per_job: dict[Job, int],
            arrival_process: FixedArrival,
            nr_of_jobs_in_system_over_time: list[int]) -> tuple[Queue, float]:
        if self.verbose:
            print(f'search with waiting area {waiting_area}')
        best_mdl_score = float('inf')
        best_queue = None
        best_c = 0
        stepsize = 1
        current_candidate_c = 1
        explored_candidates = set()
        c_upper_bound = UpperBoundEstimator(waiting_area).estimate_nr_of_servers(
            observed_arrivals, observed_departures)
        iterations_without_improvement = 0
        while current_candidate_c:
            if abs(stepsize) == 1:
                while current_candidate_c in explored_candidates and (1 < current_candidate_c < c_upper_bound - 1):
                    current_candidate_c += stepsize
            if current_candidate_c not in explored_candidates:
                if self.verbose:
                    print(f'explore candidate c={current_candidate_c}')
                best_queue_for_c, best_mdl_score_for_c = self._find_best_queue_for_c(
                    observed_arrivals, observed_departures, departure_time_per_job,
                    nr_of_jobs_in_system_over_time, arrival_process,
                    waiting_area, current_candidate_c)
                explored_candidates.add(current_candidate_c)
            if best_mdl_score_for_c < best_mdl_score:
                best_mdl_score = best_mdl_score_for_c
                best_queue = best_queue_for_c
                best_c = current_candidate_c
                iterations_without_improvement = 0
                if self.verbose:
                    print(f'improved mdl score to {best_mdl_score}')
                stepsize *= 2
            else:
                iterations_without_improvement += 1
                if stepsize == 1:
                    if iterations_without_improvement > self.__patience:
                        stepsize = -1
                        current_candidate_c = best_c
                        iterations_without_improvement = 0
                elif stepsize == -1:
                    if iterations_without_improvement > self.__patience:
                        break
                else:
                    stepsize //= 2
            if -stepsize < current_candidate_c and best_c + stepsize <= c_upper_bound:
                current_candidate_c = best_c + stepsize
        return best_queue, best_mdl_score