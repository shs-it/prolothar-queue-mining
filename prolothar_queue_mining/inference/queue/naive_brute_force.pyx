from typing import List, Tuple, Dict

from dataclasses import dataclass
from itertools import product, chain, permutations
from scipy.stats import wasserstein_distance
from scipy.stats import energy_distance
import pandas as pd
from sklearn.metrics import mean_absolute_error

from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.queue cimport Queue
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.service_time import ServiceTime
from prolothar_queue_mining.model.service_time import ServiceTimeWithDistribution
from prolothar_queue_mining.model.waiting_area import FastFirstComeFirstServeWaitingArea
from prolothar_queue_mining.model.waiting_area import FastLastComeFirstServeWaitingArea
from prolothar_queue_mining.model.waiting_area import PriorityClassWaitingArea
from prolothar_queue_mining.model.arrival_process import FixedArrival
from prolothar_queue_mining.model.population import ListPopulation
from prolothar_queue_mining.model.observer.sojourn_time import SojournTimeRecordingObserver
from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.distribution import DiscreteDistribution
from prolothar_queue_mining.model.distribution import DiscreteDegenerateDistribution

from prolothar_queue_mining.inference.queue.times cimport infer_service_times_batch
from prolothar_queue_mining.inference.queue.queue_miner import QueueMiner
from prolothar_queue_mining.inference.queue.times cimport infer_waiting_and_service_times
from prolothar_queue_mining.inference.queue.utils import generate_distribution_candidates

class Record:
    """
    recording of quality score dependent on the input parameter space
    """
    waiting_area: str
    batch_size_distribution: DiscreteDistribution
    nr_of_servers: int
    service_time: ServiceTime
    wasserstein_distance: float
    energy_distance: float
    mean_absolute_error: float

class NaiveBruteForce(QueueMiner):
    """
    infers a queue by creating many different models which are fitted by MLE and
    then selects the models with the best fit
    """

    def __init__(
            self, nr_of_servers_candidates: List[int],
            batch_size_distribution_candidates: List[DiscreteDistribution] = None,
            categorical_feature_names: List[str] = None,
            seed: int|None = None, recording_enabled: bool = True,
            record_candidates: bool = False):
        self.__nr_of_servers_candidates = nr_of_servers_candidates
        if batch_size_distribution_candidates is None:
            self.__batch_size_distribution_candidates = []
        else:
            self.__batch_size_distribution_candidates = batch_size_distribution_candidates
        self.__seed = seed
        self.__recording: list[Record] = []
        self.__recording_enabled = recording_enabled
        self.__record_candidates = record_candidates
        self.__recorded_candidates: list[Queue] = []
        if categorical_feature_names is None:
            categorical_feature_names = []
        self.__categorical_feature_names = categorical_feature_names

    def infer_queue(
            self, observed_arrivals: List[Tuple[Job, int]],
            observed_departures: List[Tuple[Job, int]]) -> Queue:
        cdef list jobs_list = [job for job,_ in observed_arrivals]
        cdef list actual_arrival_times = [arrival_time for _,arrival_time in observed_arrivals]
        cdef dict departure_time_per_job = {job: departure_time for job, departure_time in observed_departures}
        cdef list actual_sojourn_times = [
            departure_time_per_job[job] - arrival_time
            for job, arrival_time in observed_arrivals
            if job in departure_time_per_job]
        cdef int max_timestep = observed_departures[-1][1]
        cdef Queue best_queue = None
        cdef Queue queue
        cdef Queue queue_copy
        cdef float best_distance = float('inf')
        cdef float distance
        for nr_of_servers, waiting_area in product(
                self.__nr_of_servers_candidates,
                self.__yield_waiting_area_candidates(jobs_list)):
            _, service_times_per_job, _ = infer_waiting_and_service_times(
                observed_arrivals, observed_departures, waiting_area, nr_of_servers)
            batches, _, batch_service_times = infer_service_times_batch(
                observed_arrivals, observed_departures, waiting_area, nr_of_servers)

            for service_time, batch_size_distribution in chain(
                    product(
                        self.__generate_service_time_candidates(service_times_per_job),
                        [DiscreteDegenerateDistribution(1)]
                    ),
                    product(
                        self.__generate_batch_service_time_candidates(batches, batch_service_times),
                        self.__batch_size_distribution_candidates
                    )):
                queue = Queue(
                    FixedArrival(ListPopulation(jobs_list), actual_arrival_times).copy(),
                    [Server(service_time.copy()) for _ in range(nr_of_servers)],
                    waiting_area=waiting_area.copy(),
                    batch_size_distribution=batch_size_distribution)
                queue_copy = queue.copy()
                queue.set_sojourn_time_observer(SojournTimeRecordingObserver())

                environment = Environment(verbose=False)
                queue.schedule_next_arrival(environment)
                environment.run_timesteps(max_timestep)

                if queue.get_sojourn_time_observer().get_sojourn_times():
                    distance = wasserstein_distance(
                        actual_sojourn_times, queue.get_sojourn_time_observer().get_sojourn_times())
                else:
                    distance = float('inf')

                if distance < best_distance:
                    best_queue = queue_copy
                    best_distance = distance

                if self.__recording_enabled and distance != float('inf'):
                    record = Record()
                    record.waiting_area = waiting_area.get_discipline_name()
                    record.batch_size_distribution = batch_size_distribution
                    record.nr_of_servers = nr_of_servers
                    record.service_time = service_time
                    record.wasserstein_distance = distance
                    record.energy_distance = energy_distance(
                        actual_sojourn_times,
                        queue.get_sojourn_time_observer().get_sojourn_times()
                    )
                    record.mean_absolute_error = self.__compute_mean_absolute_error(
                        actual_sojourn_times,
                        queue.get_sojourn_time_observer().get_sojourn_times()
                    )
                    self.__recording.append(record)
                if self.__record_candidates:
                    self.__recorded_candidates.append(Queue(
                        None,
                        [Server(service_time.copy()) for _ in range(nr_of_servers)],
                        waiting_area=waiting_area.copy(),
                        batch_size_distribution=batch_size_distribution.copy())
                    )
        best_queue.set_arrival_process(None)
        return best_queue

    def get_recorded_candidates(self) -> List[Queue]:
        return self.__recorded_candidates

    def __yield_waiting_area_candidates(self, jobs_list: List[Job]):
        yield FastFirstComeFirstServeWaitingArea()
        yield FastLastComeFirstServeWaitingArea()
        for feature in self.__categorical_feature_names:
            categories = set(job.features[feature] for job in jobs_list)
            for category_order in permutations(categories):
                yield PriorityClassWaitingArea(feature, category_order, FastFirstComeFirstServeWaitingArea)
                yield PriorityClassWaitingArea(feature, category_order, FastLastComeFirstServeWaitingArea)

    def __compute_mean_absolute_error(
            self, actual_departure_times: List[int],
            predicted_departure_times: List[int]):
        min_length = min(len(actual_departure_times), len(predicted_departure_times))
        if min_length == 0:
            return 0
        actual_departure_times = actual_departure_times[:min_length]
        predicted_departure_times = predicted_departure_times[:min_length]
        return mean_absolute_error(actual_departure_times, predicted_departure_times)

    def get_recording_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                (
                    recording.waiting_area,
                    recording.batch_size_distribution,
                    recording.nr_of_servers,
                    recording.service_time,
                    recording.wasserstein_distance,
                    recording.energy_distance,
                    recording.mean_absolute_error
                )
                for recording in self.__recording
            ],
            columns=['D', 'B', 'c', 'S', 'wasserstein_distance', 'energy_distance', 'MAE']
        )

    def __generate_service_time_candidates(self, service_times_per_job: Dict[Job, float]):
        for distribution in generate_distribution_candidates(
                list(service_times_per_job.values()), seed_for_distributions=self.__seed):
            yield ServiceTimeWithDistribution(distribution)

    def __generate_batch_service_time_candidates(self, batches: List[List[Job]], batch_service_times: List[float]):
        for distribution in generate_distribution_candidates(
                batch_service_times, seed_for_distributions=self.__seed):
            yield ServiceTimeWithDistribution(distribution)
