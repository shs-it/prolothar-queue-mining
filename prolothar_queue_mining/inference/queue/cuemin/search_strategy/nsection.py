from math import ceil, floor
import  numpy as np
from tqdm import tqdm

from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.arrival_process import FixedArrival
from prolothar_queue_mining.model.job import Job

from prolothar_queue_mining.inference.queue.nr_of_servers import UpperBoundEstimator

from prolothar_queue_mining.inference.queue.cuemin.search_strategy.search_strategy import SearchStrategy

class NSectionSearch(SearchStrategy):

    def __init__(self,
        n: int,
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
            nr_of_cpus_for_sklearn=nr_of_cpus_for_sklearn,
            verbose=verbose,
            seed_for_distributions=seed_for_distributions,
            nr_of_load_clusters_candidates=nr_of_load_clusters_candidates,
            categorical_attribute_names=categorical_attribute_names,
            numerical_attribute_names=numerical_attribute_names)
        if n < 2:
            raise ValueError(f'n must not be < 2 but was {n}')
        self.__n = n

    def search(
            self, waiting_area: WaitingArea,
            observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]],
            departure_time_per_job: dict[Job, int],
            arrival_process: FixedArrival,
            nr_of_jobs_in_system_over_time: list[int]) -> tuple[Queue, float]:
        min_c = 1
        max_c = UpperBoundEstimator(waiting_area).estimate_nr_of_servers(observed_arrivals, observed_departures)

        c_candidates = [min_c, max_c]
        for i in range(1, self.__n - 1):
            c_candidates.append(floor((min_c + max_c) * (i / (self.__n - 1))))
        c_candidates = sorted(set(c_candidates))
        mdl_of_candidates = []
        queues_of_candidates = []
        for c in c_candidates:
            min_c_queue, min_c_score = self._find_best_queue_for_c(
                observed_arrivals, observed_departures, departure_time_per_job,
                nr_of_jobs_in_system_over_time, arrival_process,
                waiting_area, c)
            mdl_of_candidates.append(min_c_score)
            queues_of_candidates.append(min_c_queue)
        index_of_min = np.argmin(mdl_of_candidates)
        index_of_max = np.argmax(mdl_of_candidates)
        c_candidate_range = c_candidates[-1] - c_candidates[0] + 1

        with tqdm(total=c_candidate_range, leave=False,
                  desc='remaining c', disable=not self.verbose,
                  bar_format='{desc}: {n_fmt}/{total_fmt}') as progress_bar:
            while len(c_candidates) > 1:
                if index_of_max > index_of_min:
                    index_of_neighbor = index_of_max - 1
                    new_c_for_max_index = (c_candidates[index_of_max] + c_candidates[index_of_neighbor]) // 2
                    corrector_for_index_of_min = 0
                else:
                    index_of_neighbor = index_of_max + 1
                    corrector_for_index_of_min = 1
                    new_c_for_max_index = ceil((c_candidates[index_of_max + 1] + c_candidates[index_of_max]) / 2)
                if new_c_for_max_index != c_candidates[index_of_neighbor]:
                    c_candidates[index_of_max] = new_c_for_max_index
                    min_c_queue, min_c_score = self._find_best_queue_for_c(
                        observed_arrivals, observed_departures, departure_time_per_job,
                        nr_of_jobs_in_system_over_time, arrival_process,
                        waiting_area, new_c_for_max_index)
                    mdl_of_candidates[index_of_max] = min_c_score
                    queues_of_candidates[index_of_max] = min_c_queue
                    if min_c_score < mdl_of_candidates[index_of_min]:
                        index_of_min = index_of_max
                else:
                    del mdl_of_candidates[index_of_max]
                    del queues_of_candidates[index_of_max]
                    del c_candidates[index_of_max]
                    index_of_min -= corrector_for_index_of_min
                index_of_max = np.argmax(mdl_of_candidates)

                progress_bar.n = c_candidate_range - (c_candidates[-1] - c_candidates[0])
                progress_bar.update(0)

        return queues_of_candidates[0], mdl_of_candidates[0]
