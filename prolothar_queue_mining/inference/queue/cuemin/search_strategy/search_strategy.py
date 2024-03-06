from abc import ABC, abstractmethod

from collections import defaultdict
from itertools import pairwise, product, chain

from math import log2, sqrt
from prolothar_common.experiments.statistics import Statistics
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from tqdm import tqdm

from prolothar_common.mdl_utils import L_N
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.service_time import ServiceTime
from prolothar_queue_mining.model.service_time import ServiceTimeWithDistribution
from prolothar_queue_mining.model.service_time import ServiceTimeWithRegressor
from prolothar_queue_mining.model.service_time.load_dependent_service_time import LoadDependentServiceTime
from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.distribution import DiscreteDegenerateDistribution
from prolothar_queue_mining.model.distribution import PoissonDistribution
from prolothar_queue_mining.model.distribution import GeometricDistribution
from prolothar_queue_mining.model.distribution import Distribution
from prolothar_queue_mining.model.distribution import NormalDistribution
from prolothar_queue_mining.model.distribution import C2dDistribution
from prolothar_queue_mining.model.distribution.two_sided_geometric import TwoSidedGeometricDistribution
from prolothar_queue_mining.model.job.regressor import SklearnRegressor
from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.arrival_process import FixedArrival
from prolothar_queue_mining.model.job import Job

from prolothar_queue_mining.inference.sklearn.job_regression import train_lasso_cv
from prolothar_queue_mining.inference.sklearn.job_regression import train_gamma_regression_cv

from prolothar_queue_mining.inference.queue.times import infer_waiting_and_service_times
from prolothar_queue_mining.inference.queue.times import infer_service_times_batch
from prolothar_queue_mining.inference.queue.utils import generate_distribution_candidates

from prolothar_queue_mining.inference.queue.cuemin.mdl_service_time import MdlServiceTime
from prolothar_queue_mining.inference.queue.cuemin.mdl_batch_size_distribution import MdlBatchSizeDistribution
from prolothar_queue_mining.inference.queue.cuemin.record import Record
from prolothar_queue_mining.inference.queue.cuemin.mdl_score import compute_length_of_model
from prolothar_queue_mining.inference.queue.cuemin.mdl_score import compute_lower_bound_implied_by_model
from prolothar_queue_mining.inference.queue.cuemin.mdl_score import compute_lower_bound_implied_by_model_and_data


class SearchStrategy(ABC):
    """
    determines how to search for the best number of servers for a given
    waiting area.
    """

    def __init__(
        self, recording_enabled: bool = True,
        record_candidates: bool = False,
        verbose: bool = False,
        seed_for_distributions: int = None,
        nr_of_cpus_for_sklearn: int = 1,
        nr_of_load_clusters_candidates: list[int] = None,
        categorical_attribute_names: list[str] = None,
        numerical_attribute_names: list[str] = None):
        self.__recording: list[Record] = []
        self.__recording_enabled = recording_enabled
        self.__record_candidates = record_candidates
        self.__recorded_candidates: list[Queue] = []
        self.verbose = verbose
        self.__seed_for_distributions = seed_for_distributions
        # (mean, stddev) => C2dDistribution(NormalDistribution(mean, stddev))
        self.__normal_error_distribution_cache: dict[tuple[int, int], C2dDistribution] = {}
        if nr_of_load_clusters_candidates is None:
            self.__nr_of_load_clusters_candidates = [2,3]
        else:
            self.__nr_of_load_clusters_candidates = nr_of_load_clusters_candidates
        if numerical_attribute_names is None:
            self.__numerical_attribute_names = []
        else:
            self.__numerical_attribute_names = numerical_attribute_names
        if categorical_attribute_names is None:
            self.__categorical_attribute_names = []
        else:
            self.__categorical_attribute_names = categorical_attribute_names
        self.__nr_of_cpus_for_sklearn = nr_of_cpus_for_sklearn
        self.__nr_of_categorical_attributes = len(self.__categorical_attribute_names)

    @abstractmethod
    def search(
        self, waiting_area: WaitingArea,
        observed_arrivals: list[tuple[Job, int]],
        observed_departures: list[tuple[Job, int]],
        departure_time_per_job: dict[Job, int],
        arrival_process: FixedArrival,
        nr_of_jobs_in_system_over_time: list[int]) -> tuple[Queue, float]:
        """
        returns the best found queue together with its mdl score for the given
        waiting area and observed data
        """

    def _find_best_queue_for_c(
            self, observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]],
            departure_time_per_job: dict[Job, int],
            nr_of_jobs_in_system_over_time: list[int],
            arrival_process: FixedArrival,
            waiting_area: WaitingArea,
            nr_of_servers: int) -> tuple[Queue, float]:
        best_mdl_score = float('inf')
        best_queue = None
        _, service_times_per_job, _ = infer_waiting_and_service_times(
            observed_arrivals, observed_departures, waiting_area, nr_of_servers)
        batches, _, batch_service_times = infer_service_times_batch(
            observed_arrivals, observed_departures, waiting_area, nr_of_servers)
        observed_batch_sizes = [len(b) for b in batches]
        no_batching_service_time_histogram = self.__create_histogram(service_times_per_job.values())
        batching_service_time_histogram = self.__create_histogram(batch_service_times)
        batch_size_histogram = self.__create_histogram(observed_batch_sizes)
        nr_of_jobs = len(departure_time_per_job)

        for service_time, batch_size_distribution in tqdm(chain(
                product(
                    self.__generate_service_time_candidates(
                        departure_time_per_job, service_times_per_job,
                        nr_of_jobs_in_system_over_time),
                    [MdlBatchSizeDistribution(DiscreteDegenerateDistribution(1), observed_batch_sizes)]
                ),
                product(
                    self.__generate_batch_service_time_candidates(batches, batch_service_times),
                    self.__generate_batch_size_distribution_candidates(observed_batch_sizes)
                )), disable=not self.verbose, desc='S,B', leave=False):
            if compute_lower_bound_implied_by_model(
                    waiting_area, nr_of_servers, service_time,
                    batch_size_distribution.get_distribution(),
                    nr_of_jobs,
                    self.__nr_of_categorical_attributes) >= best_mdl_score:
                continue
            if compute_lower_bound_implied_by_model_and_data(
                    waiting_area, nr_of_servers, service_time,
                    batch_size_distribution.get_distribution(),
                    no_batching_service_time_histogram, batching_service_time_histogram,
                    batch_size_histogram, self.__nr_of_categorical_attributes) >= best_mdl_score:
                continue
            candidate_mdl_score = self.__run_candidate_model(
                arrival_process, waiting_area, nr_of_servers, service_time,
                batch_size_distribution, departure_time_per_job)

            if candidate_mdl_score < best_mdl_score:
                best_queue = Queue(
                    None,
                    [Server(service_time) for _ in range(nr_of_servers)],
                    waiting_area=waiting_area,
                    batch_size_distribution=batch_size_distribution.get_distribution())
                best_mdl_score = candidate_mdl_score
        return best_queue, best_mdl_score

    def __create_histogram(self, values):
        histogram = defaultdict(int)
        for v in values:
            histogram[v] += 1
        return histogram

    def __run_candidate_model(
            self, arrival_process: FixedArrival, waiting_area: WaitingArea,
            nr_of_servers: int, service_time: ServiceTime,
            batch_size_distribution: MdlBatchSizeDistribution,
            departure_time_per_job: dict[Job, int]) -> float:
        environment = Environment(verbose=False)
        mdl_service_time = MdlServiceTime(environment, service_time.copy(), departure_time_per_job)
        batch_size_distribution = batch_size_distribution.copy()
        queue = Queue(
            arrival_process.copy(),
            [Server(mdl_service_time) for _ in range(nr_of_servers)],
            waiting_area=waiting_area.copy(),
            batch_size_distribution=batch_size_distribution)
        queue.schedule_next_arrival(environment)
        environment.run_until_event_queue_is_empty()
        while queue.get_waiting_area().has_next_job():
            mdl_service_time.get_service_time(
                queue.get_waiting_area().pop_next_job(len(queue.get_waiting_area())),
                len(queue.get_waiting_area()))

        candidate_mdl_score_model = compute_length_of_model(
            queue.get_waiting_area(), nr_of_servers, service_time,
            batch_size_distribution.get_distribution(), len(self.__categorical_attribute_names))
        candidate_mdl_score_service_time = mdl_service_time.get_total_encoded_length()
        candidate_mdl_score_batch_distribution = batch_size_distribution.get_total_encoded_length()
        candidate_mdl_score = (
            candidate_mdl_score_model +
            candidate_mdl_score_service_time +
            candidate_mdl_score_batch_distribution
        )

        if self.__recording_enabled:
            self.__recording.append(Record(
                waiting_area=waiting_area.get_discipline_name(),
                batch_size_distribution=str(batch_size_distribution.get_distribution()),
                nr_of_servers=nr_of_servers,
                service_time=service_time,
                mdl_model=candidate_mdl_score_model,
                mdl_service_time=candidate_mdl_score_service_time,
                mdl_service_time_values=mdl_service_time.get_total_length_of_value_codes(),
                mdl_service_time_residual=mdl_service_time.get_total_length_of_residual_codes(),
                mdl_batching=candidate_mdl_score_batch_distribution,
                mdl_score=candidate_mdl_score,
            ))
        if self.__record_candidates:
            self.__recorded_candidates.append(Queue(
                None,
                [Server(service_time.copy()) for _ in range(nr_of_servers)],
                waiting_area=waiting_area.copy(),
                batch_size_distribution=batch_size_distribution.get_distribution())
            )
        return candidate_mdl_score

    def __generate_batch_size_distribution_candidates(self, observed_batch_sizes: list[int]):
        inferred_batch_size = round(DiscreteDegenerateDistribution.fit(observed_batch_sizes).get_mean())
        if inferred_batch_size > 1:
            yield MdlBatchSizeDistribution(DiscreteDegenerateDistribution(inferred_batch_size), observed_batch_sizes)

        poisson_candidate_distribution = PoissonDistribution.fit(observed_batch_sizes, seed=self.__seed_for_distributions)
        #make sure that poisson distribution is not degenerate
        if poisson_candidate_distribution.get_mean() != poisson_candidate_distribution.get_shift():
            yield MdlBatchSizeDistribution(poisson_candidate_distribution, observed_batch_sizes)

    def __generate_batch_service_time_candidates(self, batches: list[list[Job]], batch_service_times: list[float]):
        for distribution in generate_distribution_candidates(
                batch_service_times, seed_for_distributions=self.__seed_for_distributions):
            yield ServiceTimeWithDistribution(distribution)

    def __generate_service_time_candidates(
            self,
            departure_time_per_job: dict[Job, int],
            service_times_per_job: dict[Job, int],
            number_of_jobs_in_system_over_time: list[int]):
        yield from self.__generate_service_time_candidates_for_cluster(service_times_per_job)
        max_departure_time = max(departure_time_per_job.values())
        start_of_service_per_job = {
            job: departure_time - service_times_per_job[job]
            for job, departure_time in departure_time_per_job.items()
            if job in service_times_per_job \
            and departure_time - service_times_per_job[job] < max_departure_time
        }
        x_jobs = [job for job, _ in start_of_service_per_job.items()]
        x_system_load = np.array([
            [
                number_of_jobs_in_system_over_time[start_of_service] + 1
            ] for _, start_of_service in start_of_service_per_job.items()
        ])
        y_service_time = np.array([
            service_times_per_job[job] for job,_ in start_of_service_per_job.items()
        ])
        for k in tqdm(self.__nr_of_load_clusters_candidates, disable=not self.verbose, desc='k', leave=False):
            tree_regressor = DecisionTreeRegressor(
                max_leaf_nodes=k, random_state=self.__seed_for_distributions)
            tree_regressor.fit(x_system_load, y_service_time)
            load_threshold_list = sorted(int(t) for t in tree_regressor.tree_.threshold if t >= 1)
            if k != len(load_threshold_list) + 1:
                break
            cluster_list = self.__create_cluster_list(
                service_times_per_job, x_jobs, x_system_load, load_threshold_list)
            service_time_list = [
                self.__select_servicetime_submodel(cluster)
                for cluster in cluster_list
            ]
            service_time_list_names = [repr(s) for s in service_time_list]
            #the submodels should really be necessary, i.e.
            # there should be more than one unique submodel
            # and neighbored submodels should also differ from each other
            if len(set(service_time_list_names)) > 1 \
            and all(s1 != s2 for s1,s2 in pairwise(service_time_list_names)):
                yield LoadDependentServiceTime(service_time_list, load_threshold_list)

    def __select_servicetime_submodel(
            self, service_time_per_job_of_cluster: dict[Job, int]):
        best_service_time_submodel = None
        best_score = float('inf')
        for service_time_submodel in self.__generate_service_time_candidates_for_cluster(
                service_time_per_job_of_cluster):
            candidate_score = self.__compute_servicetime_submodel_score(
                service_time_per_job_of_cluster, service_time_submodel)
            if candidate_score < best_score:
                best_service_time_submodel = service_time_submodel
                best_score = candidate_score
        if best_service_time_submodel is None:
            #all observed service times were negative
            return ServiceTimeWithDistribution(DiscreteDegenerateDistribution(1))
        return best_service_time_submodel

    def __compute_servicetime_submodel_score(
            self, service_time_per_job_of_cluster: dict[Job, int],
            service_time_submodel: ServiceTime) -> float:
        score = service_time_submodel.get_mdl_of_model()
        for job, required_service_time in service_time_per_job_of_cluster.items():
            probability = service_time_submodel.compute_probability(required_service_time, job, None)
            if probability > Distribution.ALLMOST_ZERO:
                score += -log2(probability)
            else:
                score += L_N(abs(required_service_time) + 1)
        return score

    def __create_cluster_list(self, service_times_per_job, x_jobs, x_system_load, load_threshold_list):
        cluster_list = [{} for _ in range(len(load_threshold_list) + 1)]
        last_cluster = cluster_list[-1]
        for job, system_load in zip(x_jobs, x_system_load):
            for i, threshold in enumerate(load_threshold_list):
                if system_load <= threshold:
                    cluster_list[i][job] = service_times_per_job[job]
                    break
            else:
                last_cluster[job] = service_times_per_job[job]
        return cluster_list

    def __generate_service_time_candidates_for_cluster(self, service_times_per_job: dict[Job, int]):
        service_time_list = list(service_times_per_job.values())
        for distribution in generate_distribution_candidates(
                service_time_list,
                seed_for_distributions=self.__seed_for_distributions):
            yield ServiceTimeWithDistribution(distribution)

        #cross validation does only make sense for large enough dataset
        if (self.__categorical_attribute_names or self.__numerical_attribute_names) \
        and len(service_times_per_job) > 50 and Statistics(service_times_per_job.values()).mean() > 0:
            sklearn_model, job_to_vector_transformer, error_distribution = train_lasso_cv(
                service_times_per_job, self.__categorical_attribute_names,
                self.__numerical_attribute_names, force_positive_coefficients=True,
                nr_of_cpus=self.__nr_of_cpus_for_sklearn)
            regressor = SklearnRegressor(sklearn_model, job_to_vector_transformer, cache_enabled=True)
            original_stddev = sqrt(error_distribution.get_variance())
            for error_stddev in set(
                int(original_stddev * stddev_factor)
                for stddev_factor in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                if int(original_stddev * stddev_factor) > 0
            ):
                yield ServiceTimeWithRegressor(
                    regressor,
                    self.__create_normal_error_distribution(
                        round(error_distribution.get_mean()), error_stddev
                    )
                )

            sklearn_model, job_to_vector_transformer, error_distribution = train_gamma_regression_cv(
                {
                    job: y if y > 0 else 0.0001 for job,y in service_times_per_job.items()
                },
                self.__categorical_attribute_names,
                self.__numerical_attribute_names,
                nr_of_cpus=self.__nr_of_cpus_for_sklearn)
            regressor = SklearnRegressor(sklearn_model, job_to_vector_transformer, cache_enabled=True)
            original_left_p = error_distribution.get_negative_distribution().get_success_probability()
            original_right_p = error_distribution.get_positive_distribution().get_success_probability()
            for scaled_left_p, scaled_right_p in set(
                (
                    original_left_p + (1 - original_left_p) * scaling_factor,
                    original_right_p + (1 - original_right_p) * scaling_factor
                )
                for scaling_factor in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            ):
                yield ServiceTimeWithRegressor(
                    regressor, TwoSidedGeometricDistribution(
                        GeometricDistribution(scaled_left_p, seed=self.__seed_for_distributions),
                        GeometricDistribution(scaled_right_p, seed=self.__seed_for_distributions),
                        error_distribution.get_pmf_weights(),
                        seed=self.__seed_for_distributions
                    )
                )

    def __create_normal_error_distribution(self, error_mean: int, error_stddev: int) -> C2dDistribution:
        try:
            return self.__normal_error_distribution_cache[(error_mean, error_stddev)]
        except KeyError:
            error_distribution = C2dDistribution(NormalDistribution(
                error_mean,
                error_stddev,
                seed=self.__seed_for_distributions
            ))
            self.__normal_error_distribution_cache[(error_mean, error_stddev)] = error_distribution
            return error_distribution

    def get_recorded_candidates(self) -> list[Queue]:
        """
        the list of generated candidate models in the same order as
        get_recording_dataframe(). this list is empty by default. recording
        of candidates must be explicity enabled when calling the constructor.
        by default, this is deactivated to save memory.
        """
        return self.__recorded_candidates

    def get_recording_dataframe(self) -> pd.DataFrame:
        """
        returns a dataframe with the generated model candidates during search
        and their corresponding MDL scores.

        this dataframe is non-empty iff recording has not been deactivated in
        the constructor. by default, recording is enabled.
        """
        return pd.DataFrame(
            [
                (
                    recording.waiting_area,
                    recording.batch_size_distribution,
                    recording.nr_of_servers,
                    recording.service_time,
                    recording.mdl_model,
                    recording.mdl_batching,
                    recording.mdl_service_time,
                    recording.mdl_service_time_values,
                    recording.mdl_service_time_residual,
                    recording.mdl_score
                )
                for recording in self.__recording
            ],
            columns=['D', 'B', 'c', 'S', 'L(M)', 'L(D|B)', 'L(D|S)', 'L(D|V_S)', 'L(D|R_S)', 'mdl_score']
        )