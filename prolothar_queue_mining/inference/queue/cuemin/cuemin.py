import sys

import pandas as pd

from tqdm import tqdm

from prolothar_queue_mining.inference.queue.cuemin.search_strategy import LinearSearch
from prolothar_queue_mining.inference.queue.cuemin.search_strategy import NSectionSearch
from prolothar_queue_mining.inference.queue.cuemin.search_strategy import AdaptiveStepSizeSearch
from prolothar_queue_mining.inference.queue.cuemin.search_strategy import SimulatedAnnealing
from prolothar_queue_mining.inference.queue.cuemin.search_strategy import WeightedSampling

from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.waiting_area import FastFirstComeFirstServeWaitingArea
from prolothar_queue_mining.model.waiting_area import FastLastComeFirstServeWaitingArea
from prolothar_queue_mining.model.waiting_area import RandomOrderWaitingArea
from prolothar_queue_mining.model.arrival_process import FixedArrival

from prolothar_queue_mining.inference.queue.waiting_area import PriorityClassWaitingAreaEstimator
from prolothar_queue_mining.inference.queue.waiting_area import LinearRegressionEstimator
from prolothar_queue_mining.inference.queue.waiting_area import FlifoWaitingAreaEstimator
from prolothar_queue_mining.inference.queue.queue_miner import QueueMiner
from prolothar_queue_mining.inference.queue.utils import count_nr_of_jobs_in_system

class CueMin(QueueMiner):
    """
    infers a queue using a MDL based selection criterion
    """

    def __init__(
        self, seed_for_distributions: int = None,
        categorical_attribute_names: list[str] = None,
        numerical_attribute_names: list[str] = None,
        patience: int = sys.maxsize,
        recording_enabled: bool = True,
        record_candidates: bool = False,
        nr_of_load_clusters_candidates: list[int] = None,
        search_strategy_name: str = 'adaptive',
        waiting_area_candidates: set[str] = None,
        nr_of_cpus_for_sklearn: int = 1,
        verbose: bool = False):
        """
        record_candidates

        Parameters
        ----------
        recording_enabled : bool, optional
            if True, then records score for search candidates for
            later analysis, by default True
        record_candidates : bool, optional
            if True, then saves generated candidate models in a list for
            later analysis, by default False
        seed_for_distributions : float, optional
            seed for created service time and batch size distributions during
            candidate generation, by default None
        patience : int, optional
            accepted number of iterations with increasing number of servers
            without improvement, by default sys.maxsize
        waiting_area_candidates : set[str], optional
            short names for waiting area candidates.
            can be FCFS,LCFS,LR,PQ-c,FLIFO,SIRO
            by default FCFS,LCFS,PQ-c,FLIFO
        """
        if search_strategy_name.startswith('linear'):
            if '-' not in search_strategy_name:
                min_nr_of_servers, max_nr_of_servers = None, None
            else:
                min_nr_of_servers, max_nr_of_servers = search_strategy_name.replace('linear-', '').split('-')
                min_nr_of_servers = int(min_nr_of_servers)
                max_nr_of_servers = int(max_nr_of_servers)
            self.__search_strategy = LinearSearch(
                recording_enabled=recording_enabled,
                record_candidates=record_candidates,
                patience=patience,
                nr_of_load_clusters_candidates=nr_of_load_clusters_candidates,
                categorical_attribute_names=categorical_attribute_names,
                numerical_attribute_names=numerical_attribute_names,
                nr_of_cpus_for_sklearn=nr_of_cpus_for_sklearn,
                seed_for_distributions=seed_for_distributions,
                min_nr_of_servers=min_nr_of_servers,
                max_nr_of_servers=max_nr_of_servers,
                verbose=verbose
            )
        elif search_strategy_name.endswith('-section'):
            self.__search_strategy = NSectionSearch(
                int(search_strategy_name.replace('-section', '')),
                recording_enabled=recording_enabled,
                record_candidates=record_candidates,
                nr_of_load_clusters_candidates=nr_of_load_clusters_candidates,
                categorical_attribute_names=categorical_attribute_names,
                numerical_attribute_names=numerical_attribute_names,
                nr_of_cpus_for_sklearn=nr_of_cpus_for_sklearn,
                seed_for_distributions=seed_for_distributions,
                verbose=verbose)
        elif search_strategy_name == 'adaptive':
            self.__search_strategy = AdaptiveStepSizeSearch(
                recording_enabled=recording_enabled,
                record_candidates=record_candidates,
                patience=patience,
                nr_of_load_clusters_candidates=nr_of_load_clusters_candidates,
                categorical_attribute_names=categorical_attribute_names,
                numerical_attribute_names=numerical_attribute_names,
                nr_of_cpus_for_sklearn=nr_of_cpus_for_sklearn,
                seed_for_distributions=seed_for_distributions,
                verbose=verbose)
        elif search_strategy_name.startswith('sa-'):
            self.__search_strategy = SimulatedAnnealing(
                nr_of_iterations=int(search_strategy_name.replace('sa-', '')),
                recording_enabled=recording_enabled,
                record_candidates=record_candidates,
                nr_of_load_clusters_candidates=nr_of_load_clusters_candidates,
                categorical_attribute_names=categorical_attribute_names,
                numerical_attribute_names=numerical_attribute_names,
                nr_of_cpus_for_sklearn=nr_of_cpus_for_sklearn,
                seed_for_distributions=seed_for_distributions,
                verbose=verbose)
        elif search_strategy_name == 'weighted_sampling':
            self.__search_strategy = WeightedSampling(
                patience=patience,
                recording_enabled=recording_enabled,
                record_candidates=record_candidates,
                nr_of_load_clusters_candidates=nr_of_load_clusters_candidates,
                categorical_attribute_names=categorical_attribute_names,
                numerical_attribute_names=numerical_attribute_names,
                nr_of_cpus_for_sklearn=nr_of_cpus_for_sklearn,
                seed_for_distributions=seed_for_distributions,
                verbose=verbose)
        else:
            raise ValueError(f'unknown search strategy: {search_strategy_name}')
        if categorical_attribute_names is None:
            self.__categorical_attribute_names = []
        else:
            self.__categorical_attribute_names = categorical_attribute_names
        if numerical_attribute_names is None:
            self.__numerical_attribute_names = []
        else:
            self.__numerical_attribute_names = numerical_attribute_names
        if waiting_area_candidates:
            self.__waiting_area_candidates = waiting_area_candidates
        else:
            self.__waiting_area_candidates = set(['FCFS', 'LCFS', 'PQ-c', 'FLIFO'])
        self.__verbose = verbose
        self.__seed = seed_for_distributions

    def infer_queue(
            self, observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]]) -> Queue:
        arrival_process = FixedArrival.create_from_observation(observed_arrivals)
        departure_time_per_job = dict(observed_departures)
        _,nr_of_jobs_in_system_over_time = count_nr_of_jobs_in_system(
            dict(observed_arrivals), departure_time_per_job)
        best_mdl_score = float('inf')
        best_queue = None
        for waiting_area in tqdm(list(self.__generate_waiting_area_candidates(
                observed_arrivals, observed_departures)), disable=not self.__verbose,
                desc='D'):
            best_mdl_score, best_queue = self.__search_with_given_waiting_area(
                waiting_area, observed_arrivals, observed_departures,
                departure_time_per_job, arrival_process,
                nr_of_jobs_in_system_over_time,
                best_mdl_score, best_queue)
        return best_queue

    def __search_with_given_waiting_area(
            self, waiting_area: WaitingArea,
            observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]],
            departure_time_per_job: dict[Job, int],
            arrival_process: FixedArrival,
            nr_of_jobs_in_system_over_time: list[int],
            global_best_mdl_score: float,
            global_best_queue: Queue):
        best_queue, best_mdl_score = self.__search_strategy.search(
            waiting_area, observed_arrivals, observed_departures,
            departure_time_per_job, arrival_process, nr_of_jobs_in_system_over_time)
        if best_mdl_score < global_best_mdl_score:
            return best_mdl_score, best_queue
        else:
            return global_best_mdl_score, global_best_queue

    def get_recording_dataframe(self) -> pd.DataFrame:
        """
        returns a dataframe with the generated model candidates during search
        and their corresponding MDL scores.

        this dataframe is non-empty iff recording has not been deactivated in
        the constructor. by default, recording is enabled.
        """
        return self.__search_strategy.get_recording_dataframe()

    def get_recorded_candidates(self) -> list[Queue]:
        """
        the list of generated candidate models in the same order as
        get_recording_dataframe(). this list is empty by default. recording
        of candidates must be explicity enabled when calling the constructor.
        by default, this is deactivated to save memory.
        """
        return self.__search_strategy.get_recorded_candidates()

    def __generate_waiting_area_candidates(
            self,
            observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]]):
        if 'FCFS' in self.__waiting_area_candidates:
            yield FastFirstComeFirstServeWaitingArea()
        if 'LCFS' in self.__waiting_area_candidates:
            yield FastLastComeFirstServeWaitingArea()
        if 'SIRO' in self.__waiting_area_candidates:
            yield RandomOrderWaitingArea()
        if 'FLIFO' in self.__waiting_area_candidates:
            flifo_waiting_area = FlifoWaitingAreaEstimator().infer_waiting_area(
                observed_arrivals, observed_departures)
            if flifo_waiting_area is not None:
                yield flifo_waiting_area
        if self.__categorical_attribute_names:
            if 'PQ-c' in self.__waiting_area_candidates:
                priority_class_waiting_area = PriorityClassWaitingAreaEstimator(
                    self.__categorical_attribute_names).infer_waiting_area(
                        observed_arrivals, observed_departures)
                if priority_class_waiting_area is not None:
                    yield priority_class_waiting_area
        if 'LR' in self.__waiting_area_candidates \
        and (self.__categorical_attribute_names or self.__numerical_attribute_names):
            yield LinearRegressionEstimator(
                self.__numerical_attribute_names,
                self.__categorical_attribute_names,
                seed=self.__seed
            ).infer_waiting_area(observed_arrivals, observed_departures)
