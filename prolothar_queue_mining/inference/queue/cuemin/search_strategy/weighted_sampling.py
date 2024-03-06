from tabnanny import verbose
import numpy as np
from tqdm import trange

from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.arrival_process import FixedArrival
from prolothar_queue_mining.model.job import Job

from prolothar_queue_mining.inference.queue.nr_of_servers import UpperBoundEstimator

from prolothar_queue_mining.inference.queue.cuemin.search_strategy.search_strategy import SearchStrategy

class WeightedSampling(SearchStrategy):

    def __init__(self,
        patience: int = 100,
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
        self.__patience = patience
        self.__random_generator = np.random.default_rng(seed_for_distributions)

    def search(
            self, waiting_area: WaitingArea,
            observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]],
            departure_time_per_job: dict[Job, int],
            arrival_process: FixedArrival,
            nr_of_jobs_in_system_over_time: list[int]) -> tuple[Queue, float]:
        iterations_without_improvement = 0
        best_queue, best_mdl_score = self._find_best_queue_for_c(
            observed_arrivals, observed_departures, departure_time_per_job,
            nr_of_jobs_in_system_over_time, arrival_process,
            waiting_area, 1)
        max_c = UpperBoundEstimator(waiting_area).estimate_nr_of_servers(observed_arrivals, observed_departures)
        candidates = np.array(list(range(2, max_c + 1)))
        candidate_scores = np.array([best_mdl_score] + [best_mdl_score * 2 for _ in range(max_c - 2)])
        candidates_indices = list(range(len(candidate_scores)))
        while len(candidates) > 0 and iterations_without_improvement <= self.__patience:
            for iterations_without_improvement in trange(
                    min(self.__patience, len(candidates)),
                    disable=not self.verbose, desc='patience', leave=True):
                candidate_weights = candidate_scores.min() / candidate_scores
                candidate_weights = candidate_weights / candidate_weights.sum()
                current_c_index = self.__random_generator.choice(candidates_indices, p=candidate_weights)
                current_c = candidates[current_c_index]
                best_queue_for_c, best_mdl_score_for_c = self._find_best_queue_for_c(
                    observed_arrivals, observed_departures, departure_time_per_job,
                    nr_of_jobs_in_system_over_time, arrival_process,
                    waiting_area, current_c)

                if current_c_index > 0:
                    candidate_scores[current_c_index - 1] = min(
                        candidate_scores[current_c_index - 1], best_mdl_score_for_c)
                if current_c_index < len(candidates) - 1:
                    candidate_scores[current_c_index + 1] = min(
                        candidate_scores[current_c_index + 1], best_mdl_score_for_c)
                candidate_scores = np.delete(candidate_scores, current_c_index)
                candidates = np.delete(candidates, current_c_index)
                del candidates_indices[-1]

                if best_mdl_score_for_c < best_mdl_score:
                    if verbose:
                        print(f'improved mdl score from {best_mdl_score} to {best_mdl_score_for_c}, c={current_c}')
                    best_mdl_score = best_mdl_score_for_c
                    best_queue = best_queue_for_c
                    break
        return best_queue, best_mdl_score